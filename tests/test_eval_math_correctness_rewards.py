from slime.rollout.sglang_rollout import _get_eval_reward_for_logging
from slime.utils.types import Sample


def test_eval_reward_uses_ground_truth_math_correctness_when_label_has_ground_truth():
    sample = Sample(
        response="After solving, the final answer is \\boxed{204}.",
        label={"ground_truth": "204", "style": "rule"},
        reward={"teacher": -3.14},
    )

    reward = _get_eval_reward_for_logging(sample, reward_key=None)

    assert reward == 1.0


def test_eval_reward_falls_back_to_scalar_reward_when_no_ground_truth_is_available():
    sample = Sample(
        response="irrelevant",
        label="unused",
        reward=0.25,
    )

    reward = _get_eval_reward_for_logging(sample, reward_key=None)

    assert reward == 0.25


def test_eval_reward_falls_back_to_reward_key_when_reward_is_a_dict():
    sample = Sample(
        response="irrelevant",
        label={"style": "rule"},
        reward={"score": 0.5, "aux": 1.0},
    )

    reward = _get_eval_reward_for_logging(sample, reward_key="score")

    assert reward == 0.5
