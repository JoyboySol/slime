import ray

from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.checkpoint_retention import prune_checkpoint_storage
from slime.utils.misc import should_run_periodic_action


def train(args):
    configure_logger()
    release_train = args.release_train

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    if args.rebuild_train_actors:
        # actor.init() is intentionally deferred until after the first rollout
        # engines are destroyed.  Normally actor.init() publishes this config,
        # but generate() needs it earlier to split replay data by DP rank.
        train_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
        train_parallel_config = {
            "dp_size": train_world_size
            // (args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size),
            "cp_size": args.context_parallel_size,
            "vpp_size": args.virtual_pipeline_model_parallel_size or 1,
            "microbatch_group_size_per_vp_stage": 1,
        }
        ray.get(rollout_manager.set_train_parallel_config.remote(train_parallel_config))

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.rebuild_train_actors:
        # The first rollout is served directly from the HF checkpoint. Keep
        # the training actor out of GPU memory until rollout data is ready.
        actor_model.release()
    else:
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())

        # Always push actor weights to rollout once weights are loaded.
        actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.destroy_rollout_engines:
            ray.get(rollout_manager.destroy_engines.remote())
        elif args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train or args.rebuild_train_actors:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        # Rebuilding the actor requires a fresh snapshot after every rollout;
        # disk retention is handled separately below.
        should_save = release_train or args.rebuild_train_actors or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        if should_save:
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        offload_train(actor_trains)
        if args.rebuild_train_actors:
            if args.save_hf is None:
                raise ValueError("--rebuild-train-actors requires --save-hf")
            actor_model.release()
            args.load = args.save
            args.ckpt_step = None
            args.finetune = False
            args.no_load_optim = args.no_save_optim
            args.no_load_rng = False
            hf_model_path = args.save_hf.format(rollout_id=rollout_id)
            ray.get(rollout_manager.recreate_engines.remote(model_path=hf_model_path))
        elif args.destroy_rollout_engines:
            ray.get(rollout_manager.recreate_engines.remote())
        elif args.offload_rollout and not release_train and rollout_id != args.num_rollout - 1:
            ray.get(rollout_manager.onload_weights.remote())

        # Keep the current HF snapshot until the recreated engines have loaded
        # it, then prune old MCore/HF snapshots according to the retention rule.
        if should_save:
            prune_checkpoint_storage(
                args.save,
                args.save_hf,
                rollout_id,
                latest_count=args.checkpoint_retention_count,
                interval=args.checkpoint_retention_interval,
            )
        # The final rollout has no subsequent generation that can consume a
        # fresh actor snapshot.  Respect the configured update interval and
        # avoid pulling the colocated SGLang engines back into memory after
        # the final checkpoint save.
        if not args.rebuild_train_actors and (args.destroy_rollout_engines or (
            (rollout_id + 1) % args.update_weights_interval == 0 and rollout_id != args.num_rollout - 1
        )):
            actor_model.update_weights()

        if args.destroy_rollout_engines:
            # Newly-created engines already own weights/KV memory; this call is
            # intentionally omitted because recreate_engines waits for init().
            pass
        elif args.offload_rollout and rollout_id != args.num_rollout - 1:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
