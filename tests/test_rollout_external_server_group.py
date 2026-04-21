from types import SimpleNamespace

from slime.ray import rollout as rollout_module


def _make_args(**overrides):
    defaults = dict(
        debug_train_only=False,
        rollout_external=True,
        num_gpus_per_node=3,
        rollout_num_gpus_per_engine=1,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        offload_rollout=False,
        hf_checkpoint="/tmp/fake-model",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_external_server_group_start_engines_does_not_require_rollout_pg_gpus(monkeypatch):
    remote_calls = []

    class FakeEngineActor:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        def remote(self, *args, **kwargs):
            remote_calls.append((self.kwargs, args, kwargs))

            class _FakeEngineHandle:
                class init:
                    @staticmethod
                    def remote(**_kwargs):
                        return ("init", _kwargs)

            return _FakeEngineHandle()

    class FakeRemote:
        @staticmethod
        def options(**kwargs):
            return FakeEngineActor(kwargs)

    monkeypatch.setattr(rollout_module.ray, "remote", lambda cls: FakeRemote)
    monkeypatch.setattr(
        rollout_module,
        "_allocate_rollout_engine_addr_and_ports_external",
        lambda args, rollout_engines: {
            rank: dict(dist_init_addr=f"127.0.0.1:{31001 + rank}", nccl_port=None, host="127.0.0.1", port=31001 + rank)
            for rank, _ in rollout_engines
        },
    )

    group = rollout_module.ServerGroup(
        args=_make_args(rollout_external=True),
        pg=(object(), [], []),
        all_engines=[None, None],
        num_gpus_per_engine=1,
        num_new_engines=0,
        worker_type="regular",
        rank_offset=0,
        gpu_offset=0,
        router_ip="127.0.0.1",
        router_port=30110,
    )

    init_handles, _ = group.start_engines()

    assert len(remote_calls) == 2
    assert all(call[0]["num_gpus"] == 0.0 for call in remote_calls)
    assert all("scheduling_strategy" not in call[0] for call in remote_calls)
    assert all(call[2]["base_gpu_id"] == 0 for call in remote_calls)
    assert len(init_handles) == 2
