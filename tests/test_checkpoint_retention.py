from pathlib import Path

from slime.utils.checkpoint_retention import prune_checkpoint_storage


def _touch_checkpoint(parent: Path, name: str) -> None:
    path = parent / name
    path.mkdir()
    (path / "marker").touch()


def test_retains_latest_two_and_every_twentieth_step(tmp_path):
    mcore = tmp_path / "mcore"
    hf = tmp_path / "hf"
    mcore.mkdir()
    hf.mkdir()
    for rollout_id in range(25):
        _touch_checkpoint(mcore, f"iter_{rollout_id:07d}")
        _touch_checkpoint(hf, f"rollout_{rollout_id}")

    prune_checkpoint_storage(
        mcore,
        str(hf / "rollout_{rollout_id}"),
        current_id=24,
        latest_count=2,
        interval=20,
    )

    # One-based steps 20 and 40 map to rollout IDs 19 and 39; only 19 exists.
    expected = {19, 23, 24}
    assert {int(path.name.removeprefix("iter_")) for path in mcore.iterdir()} == expected
    assert {int(path.name.removeprefix("rollout_")) for path in hf.iterdir()} == expected
