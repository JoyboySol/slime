from types import SimpleNamespace

from slime.utils import logging_utils


class _FakeScheduler:
    def __init__(self, values):
        self._values = values

    def get_lr(self, param_group):
        return self._values[param_group["name"]]


def test_build_lr_metrics_uses_first_param_group_only():
    optimizer = type(
        "Optimizer",
        (),
        {
            "param_groups": [
                {"name": "first"},
                {"name": "second"},
            ]
        },
    )()
    scheduler = _FakeScheduler({"first": 1e-5, "second": 2e-5})

    metrics = logging_utils.build_lr_metrics(
        optimizer=optimizer,
        opt_param_scheduler=scheduler,
        metric_prefix="train/",
    )

    assert metrics == {"train/lr": 1e-5}


def test_log_passes_aligned_step_to_wandb(monkeypatch):
    logged = {}

    monkeypatch.setattr(
        logging_utils.wandb_utils,
        "prepare_metrics_for_wandb",
        lambda _args, metrics: {
            **metrics,
            "eval/step": 30.0,
        },
    )
    monkeypatch.setattr(logging_utils.wandb, "log", lambda data, step=None: logged.update({"data": data, "step": step}))

    args = SimpleNamespace(use_wandb=True, use_tensorboard=False)
    logging_utils.log(args, {"eval/step": 5, "eval/aime": 0.1}, step_key="eval/step")

    assert logged["data"]["eval/step"] == 30.0
    assert logged["step"] == 30.0
