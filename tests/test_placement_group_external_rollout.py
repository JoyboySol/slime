from types import SimpleNamespace

from slime.ray import placement_group as placement_group_module


def _make_args(**overrides):
    defaults = dict(
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        use_critic=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=3,
        critic_num_nodes=0,
        critic_num_gpus_per_node=0,
        rollout_num_gpus=3,
        rollout_external=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_external_rollout_does_not_reserve_local_rollout_gpus(monkeypatch):
    captured = {}

    def fake_create(num_gpus):
        captured["num_gpus"] = num_gpus
        return object(), list(range(num_gpus)), list(range(num_gpus))

    monkeypatch.setattr(placement_group_module, "_create_placement_group", fake_create)

    pgs = placement_group_module.create_placement_groups(
        _make_args(
            rollout_external=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=3,
            rollout_num_gpus=3,
        )
    )

    assert captured["num_gpus"] == 3
    assert pgs["rollout"][1] == []
    assert pgs["rollout"][2] == []


def test_local_rollout_still_reserves_rollout_gpus(monkeypatch):
    captured = {}

    def fake_create(num_gpus):
        captured["num_gpus"] = num_gpus
        return object(), list(range(num_gpus)), list(range(num_gpus))

    monkeypatch.setattr(placement_group_module, "_create_placement_group", fake_create)

    pgs = placement_group_module.create_placement_groups(
        _make_args(
            rollout_external=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=3,
            rollout_num_gpus=3,
        )
    )

    assert captured["num_gpus"] == 6
    assert pgs["rollout"][1] == [3, 4, 5]
    assert pgs["rollout"][2] == [3, 4, 5]
