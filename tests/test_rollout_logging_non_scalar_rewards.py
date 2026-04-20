from argparse import Namespace

from slime.ray import rollout as rollout_module
from slime.utils.types import Sample


def make_args() -> Namespace:
    return Namespace(
        advantage_estimator="grpo",
        reward_key=None,
        log_reward_category=None,
        custom_eval_rollout_log_function_path=None,
        custom_rollout_log_function_path=None,
        load_debug_rollout_data=False,
        log_passrate=True,
        n_samples_per_eval_prompt=1,
        wandb_always_use_train_step=False,
        rollout_batch_size=2,
        n_samples_per_prompt=4,
        global_batch_size=8,
        rollout_num_gpus=1,
        sglang_speculative_algorithm=None,
    )


def make_sample(index: int, group_index: int, truncated: bool) -> Sample:
    return Sample(
        index=index,
        group_index=group_index,
        prompt=f"prompt-{index}",
        response="answer",
        response_length=4,
        reward={"meta_info": {"request_id": index}},
        status=Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED,
        non_generation_time=0.0,
    )


def test_compute_metrics_from_samples_skips_zero_std_for_non_scalar_rewards():
    args = make_args()
    samples = [make_sample(0, 0, False), make_sample(1, 1, True)]

    metrics = rollout_module.compute_metrics_from_samples(args, samples)

    assert metrics["response_len/mean"] == 4.0
    assert metrics["truncated_ratio"] == 0.5
    assert not any(key.startswith("zero_std/") for key in metrics)


def test_log_eval_rollout_data_skips_non_scalar_reward_average(monkeypatch):
    args = make_args()
    samples = [make_sample(0, 0, False), make_sample(1, 1, True)]
    logged = {}

    monkeypatch.setattr(rollout_module.logging_utils, "log", lambda _args, data, step_key: logged.update(data))

    log_dict = rollout_module._log_eval_rollout_data(
        3,
        args,
        {
            "aime": {
                "rewards": [sample.reward for sample in samples],
                "truncated": [sample.status == Sample.Status.TRUNCATED for sample in samples],
                "samples": samples,
            }
        },
        extra_metrics=None,
    )

    assert "eval/aime" not in log_dict
    assert log_dict["eval/aime-truncated_ratio"] == 0.5
    assert log_dict["eval/aime/response_len/mean"] == 4.0
    assert logged["eval/step"] == 3


def test_log_rollout_data_accepts_non_scalar_rewards(monkeypatch):
    args = make_args()
    samples = [make_sample(0, 0, False), make_sample(1, 1, True)]
    logged = {}

    monkeypatch.setattr(rollout_module.logging_utils, "log", lambda _args, data, step_key: logged.update(data))

    rollout_module._log_rollout_data(
        5,
        args,
        samples,
        rollout_extra_metrics={},
        rollout_time=2.0,
    )

    assert logged["rollout/step"] == 5
    assert logged["rollout/response_len/mean"] == 4.0
    assert logged["rollout/truncated_ratio"] == 0.5


def test_log_eval_rollout_data_logs_scalar_score_for_string_label(monkeypatch):
    args = make_args()
    samples = [
        Sample(
            index=0,
            group_index=0,
            prompt="prompt-0",
            response="The answer is \\boxed{4}",
            response_length=4,
            reward={"meta_info": {"request_id": 0}},
            label="4",
            status=Sample.Status.COMPLETED,
            non_generation_time=0.0,
        )
    ]
    logged = {}

    monkeypatch.setattr(rollout_module.logging_utils, "log", lambda _args, data, step_key: logged.update(data))

    log_dict = rollout_module._log_eval_rollout_data(
        7,
        args,
        {
            "math500_100": {
                "rewards": [1.0],
                "truncated": [False],
                "samples": samples,
            }
        },
        extra_metrics=None,
    )

    assert log_dict["eval/math500_100"] == 1.0
    assert logged["eval/math500_100"] == 1.0
    assert logged["eval/step"] == 7


def test_should_save_debug_rollout_data_uses_eval_cadence():
    args = Namespace(
        save_debug_rollout_data="/tmp/rollout_{rollout_id}.pt",
        eval_interval=5,
    )

    assert rollout_module._should_save_debug_rollout_data(args, rollout_id=0, evaluation=False) is False
    assert rollout_module._should_save_debug_rollout_data(args, rollout_id=4, evaluation=False) is True
    assert rollout_module._should_save_debug_rollout_data(args, rollout_id=0, evaluation=True) is True


def test_should_save_debug_rollout_data_returns_false_without_template():
    args = Namespace(
        save_debug_rollout_data=None,
        eval_interval=5,
    )

    assert rollout_module._should_save_debug_rollout_data(args, rollout_id=4, evaluation=False) is False
