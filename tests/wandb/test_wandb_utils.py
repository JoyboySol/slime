import json
from pathlib import Path
from types import SimpleNamespace

from slime.utils import wandb_utils


class _FakeRun:
    def __init__(self, run_id: str, run_dir: str):
        self.id = run_id
        self.dir = run_dir


class _FakeWandb:
    def __init__(self, run_ids):
        self._run_ids = iter(run_ids)
        self.init_calls = []
        self.login_calls = []
        self.finish_calls = 0
        self.log_calls = []
        self.run = None
        self.util = SimpleNamespace(generate_id=lambda: "suffix123")

    def login(self, **kwargs):
        self.login_calls.append(kwargs)

    def Settings(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def init(self, **kwargs):
        self.init_calls.append(dict(kwargs))
        run_id = kwargs.get("id") or next(self._run_ids)
        run_root = Path(kwargs["dir"]) / f"run-for-{run_id}"
        run_root.mkdir(parents=True, exist_ok=True)
        self.run = _FakeRun(run_id=run_id, run_dir=str(run_root / "files"))
        return self.run

    def finish(self):
        self.finish_calls += 1
        self.run = None

    def log(self, row):
        self.log_calls.append(dict(row))

    def define_metric(self, *_args, **_kwargs):
        return None


def _make_args(tmp_path: Path, **overrides):
    values = {
        "use_wandb": True,
        "wandb_run_id": None,
        "wandb_mode": "online",
        "wandb_key": "secret",
        "wandb_host": None,
        "wandb_team": "team",
        "wandb_project": "project",
        "wandb_group": "group",
        "wandb_random_suffix": True,
        "wandb_dir": str(tmp_path / "wandb"),
        "rank": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_init_wandb_primary_persists_identity_and_resumes_from_state(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb(run_ids=["run-abc"])
    monkeypatch.setattr(wandb_utils, "wandb", fake_wandb)

    args = _make_args(tmp_path)
    wandb_utils.init_wandb_primary(args)

    state_path = Path(args.wandb_dir) / "slime_wandb_state.json"
    persisted = json.loads(state_path.read_text(encoding="utf-8"))

    assert args.wandb_run_id == "run-abc"
    assert persisted["run_id"] == "run-abc"
    assert persisted["group"] == "group_suffix123"
    assert persisted["run_name"] == "group_suffix123-RANK_0"
    assert persisted["step_targets"] == {}
    assert "id" not in fake_wandb.init_calls[0]
    assert "resume" not in fake_wandb.init_calls[0]

    resumed_args = _make_args(tmp_path)
    wandb_utils.init_wandb_primary(resumed_args)

    assert resumed_args.wandb_run_id == "run-abc"
    assert fake_wandb.init_calls[1]["id"] == "run-abc"
    assert fake_wandb.init_calls[1]["resume"] == "must"
    assert fake_wandb.init_calls[1]["group"] == "group_suffix123"
    assert fake_wandb.init_calls[1]["name"] == "group_suffix123-RANK_0"


def test_init_wandb_primary_prefers_explicit_run_id_over_persisted_state(tmp_path, monkeypatch):
    state_dir = tmp_path / "wandb"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "slime_wandb_state.json").write_text(
        json.dumps({"run_id": "old-run", "group": "old-group", "run_name": "old-name"}),
        encoding="utf-8",
    )

    fake_wandb = _FakeWandb(run_ids=[])
    monkeypatch.setattr(wandb_utils, "wandb", fake_wandb)

    args = _make_args(tmp_path, wandb_run_id="manual-run", wandb_random_suffix=False)
    wandb_utils.init_wandb_primary(args)

    assert args.wandb_run_id == "manual-run"
    assert fake_wandb.init_calls[0]["id"] == "manual-run"
    assert fake_wandb.init_calls[0]["resume"] == "must"
    assert fake_wandb.init_calls[0]["group"] == "group"
    assert fake_wandb.init_calls[0]["name"] == "group"


def test_init_wandb_primary_uses_resume_from_step_with_explicit_run_id(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb(run_ids=[])
    monkeypatch.setattr(wandb_utils, "wandb", fake_wandb)

    args = _make_args(
        tmp_path,
        wandb_run_id="manual-run",
        wandb_resume_from_step=57,
        wandb_random_suffix=False,
    )
    wandb_utils.init_wandb_primary(args)

    assert args.wandb_run_id == "manual-run"
    assert fake_wandb.init_calls[0]["id"] == "manual-run"
    assert fake_wandb.init_calls[0]["resume"] == "must"


def test_init_wandb_primary_can_start_fresh_run_while_resuming_training(tmp_path, monkeypatch):
    state_dir = tmp_path / "wandb"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "slime_wandb_state.json").write_text(
        json.dumps(
            {
                "run_id": "old-run",
                "group": "old-group",
                "run_name": "old-name",
                "step_targets": {"train/step": 12},
            }
        ),
        encoding="utf-8",
    )

    fake_wandb = _FakeWandb(run_ids=["fresh-run"])
    monkeypatch.setattr(wandb_utils, "wandb", fake_wandb)

    args = _make_args(tmp_path, wandb_resume_from_step=57, wandb_start_fresh=True)
    wandb_utils.init_wandb_primary(args)

    state_path = Path(args.wandb_dir) / "slime_wandb_state.json"
    persisted = json.loads(state_path.read_text(encoding="utf-8"))

    assert args.wandb_run_id == "fresh-run"
    assert "id" not in fake_wandb.init_calls[0]
    assert "resume" not in fake_wandb.init_calls[0]
    assert fake_wandb.init_calls[0]["group"] == "group_suffix123"
    assert fake_wandb.init_calls[0]["name"] == "group_suffix123-RANK_0"
    assert persisted["run_id"] == "fresh-run"
    assert persisted["group"] == "group_suffix123"
    assert persisted["run_name"] == "group_suffix123-RANK_0"
    assert persisted["step_targets"] == {}


def test_init_wandb_primary_backfills_safe_history_into_fresh_resume_run(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb(run_ids=["fresh-run"])
    monkeypatch.setattr(wandb_utils, "wandb", fake_wandb)
    monkeypatch.setattr(
        wandb_utils,
        "_load_wandb_backfill_rows",
        lambda args, source_run_id, max_step: [
            {"train/step": 0, "train/loss": 1.0},
            {"train/step": 57, "train/loss": 0.5},
        ],
    )

    args = _make_args(
        tmp_path,
        wandb_run_id="old-run",
        wandb_resume_from_step=57,
        wandb_start_fresh=True,
    )
    wandb_utils.init_wandb_primary(args)

    assert args.wandb_run_id == "fresh-run"
    assert "id" not in fake_wandb.init_calls[0]
    assert fake_wandb.log_calls == [
        {"train/step": 0, "train/loss": 1.0},
        {"train/step": 57, "train/loss": 0.5},
    ]


def test_prepare_metrics_for_wandb_uses_persisted_step_targets(tmp_path, monkeypatch):
    state_dir = tmp_path / "wandb"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "slime_wandb_state.json").write_text(
        json.dumps(
            {
                "run_id": "canonical-run",
                "group": "canonical-group",
                "run_name": "canonical-name",
                "step_targets": {"train/step": 12, "rollout/step": 20},
            }
        ),
        encoding="utf-8",
    )

    fake_wandb = _FakeWandb(run_ids=[])
    monkeypatch.setattr(wandb_utils, "wandb", fake_wandb)

    args = _make_args(tmp_path)
    wandb_utils.init_wandb_primary(args)

    first = wandb_utils.prepare_metrics_for_wandb(args, {"train/step": 7, "rollout/step": 10, "x": 1.0})
    second = wandb_utils.prepare_metrics_for_wandb(args, {"train/step": 8, "rollout/step": 11, "x": 2.0})

    assert first["train/step"] == 12.0
    assert first["rollout/step"] == 20.0
    assert second["train/step"] == 13.0
    assert second["rollout/step"] == 21.0
