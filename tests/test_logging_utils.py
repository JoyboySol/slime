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
