from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "merge_wandb_runs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("merge_wandb_runs", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_checkpoint_safe_step_defaults_to_actor_checkpoint_file(tmp_path):
    module = _load_module()
    wandb_dir = tmp_path / "wandb"
    actor_ckpt_dir = tmp_path / "actor_ckpt"
    wandb_dir.mkdir()
    actor_ckpt_dir.mkdir()
    (actor_ckpt_dir / "latest_checkpointed_iteration.txt").write_text("34\n", encoding="utf-8")

    args = SimpleNamespace(max_safe_step=None, wandb_dir=str(wandb_dir))

    assert module._resolve_checkpoint_safe_step(args) == 34.0


def test_load_segments_prefers_local_wandb_files_in_auto_mode(monkeypatch):
    module = _load_module()
    run_paths = ["entity/project/zvpis60a"]

    monkeypatch.setattr(module, "_collect_local_wandb_files", lambda wandb_dir, run_id: [Path("/tmp/fake1.wandb")])
    monkeypatch.setattr(
        module,
        "_extract_history_rows_from_local_wandb_file",
        lambda path: [{"train/step": 30}, {"train/step": 30}, {"train/step": 31}],
    )
    monkeypatch.setattr(module, "_load_segments_from_api", lambda api, run_paths: (_ for _ in ()).throw(AssertionError("api should not be used")))

    segments, source_used = module._load_segments(
        api=object(),
        run_paths=run_paths,
        history_source="auto",
        wandb_dir="/tmp/fake-wandb-dir",
    )

    assert source_used == "local"
    assert len(segments) == 1
    assert segments[0].run_path == run_paths[0]
    assert segments[0].rows == [{"train/step": 30}, {"train/step": 31}]
