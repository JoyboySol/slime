import logging

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def _log_driver_rollout_payload(payload):
    if not payload:
        return
    summary_text = payload.get("summary_text")
    if not summary_text:
        return
    kind = payload.get("kind", "unknown")
    rollout_id = payload.get("rollout_id")
    logger.info("driver %s opd summary %s: %s", kind, rollout_id, summary_text)


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(rollout_id):
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            if args.critic_train_only:
                critic_model.clear_memory()
            else:
                actor_model.clear_memory()

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps and not args.critic_train_only):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_result = ray.get(rollout_manager.generate.remote(rollout_id))
        if isinstance(rollout_result, dict) and "rollout_data" in rollout_result:
            _log_driver_rollout_payload(rollout_result.get("driver_log_payload"))
            rollout_data_ref = rollout_result["rollout_data"]
        else:
            rollout_data_ref = rollout_result

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                actor_train_result = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
                for payload in actor_train_result:
                    _log_driver_rollout_payload(payload)
            critic_train_result = ray.get(critic_train_handle)
            for payload in critic_train_result:
                _log_driver_rollout_payload(payload)
        else:
            actor_train_result = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            for payload in actor_train_result:
                _log_driver_rollout_payload(payload)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train(rollout_id)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        if not args.critic_train_only:
            actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
