from pathlib import Path

from slime.utils.load_path_utils import resolve_bridge_initial_load_path


def test_resolve_bridge_initial_load_path_falls_back_from_empty_workdir(tmp_path: Path):
    load_dir = tmp_path / "actor_ckpt"
    load_dir.mkdir()
    ref_dir = tmp_path / "ref_load"
    ref_dir.mkdir()
    (ref_dir / "latest_checkpointed_iteration.txt").write_text("release")
    hf_dir = tmp_path / "hf_model"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")

    resolved = resolve_bridge_initial_load_path(str(load_dir), str(ref_dir), str(hf_dir))

    assert resolved == str(ref_dir)


def test_resolve_bridge_initial_load_path_keeps_explicit_hf_checkpoint(tmp_path: Path):
    hf_dir = tmp_path / "hf_model"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")
    ref_dir = tmp_path / "ref_load"
    ref_dir.mkdir()
    (ref_dir / "latest_checkpointed_iteration.txt").write_text("release")

    resolved = resolve_bridge_initial_load_path(str(hf_dir), str(ref_dir), str(hf_dir))

    assert resolved == str(hf_dir)


def test_resolve_bridge_initial_load_path_keeps_explicit_megatron_checkpoint(tmp_path: Path):
    ckpt_dir = tmp_path / "actor_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "latest_checkpointed_iteration.txt").write_text("release")
    hf_dir = tmp_path / "hf_model"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")

    resolved = resolve_bridge_initial_load_path(str(ckpt_dir), None, str(hf_dir))

    assert resolved == str(ckpt_dir)
