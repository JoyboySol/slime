import asyncio
import json
from types import SimpleNamespace

from yongxiang.bigcodebench_rm import compute_score, reward


TEST_CODE = """
import unittest

class TestCases(unittest.TestCase):
    def test_add_one(self):
        self.assertEqual(task_func(4), 5)
"""


def _label(test_code: str = TEST_CODE) -> str:
    return json.dumps({
        "task_id": "BigCodeBench/test",
        "entry_point": "task_func",
        "test": test_code,
        "libs": [],
    })


def test_correct_solution_passes():
    result = compute_score("```python\ndef task_func(value):\n    return value + 1\n```", _label())
    assert result["acc"] is True
    assert result["status"] == "passed"


def test_incorrect_solution_runs_tests_and_fails():
    result = compute_score("def task_func(value):\n    return value", _label())
    assert result["acc"] is False
    assert result["status"] == "failed"
    assert "FAILED" in result["stderr"]


def test_reasoning_and_generation_markers_are_removed():
    response = "<think>reasoning</think>\n```python\ndef task_func(value):\n    return value + 1\n```\n<|im_end|>"
    assert compute_score(response, _label())["acc"] is True


def test_no_tests_is_not_a_pass():
    result = compute_score("def task_func(value):\n    return value", _label("import unittest\n"))
    assert result["acc"] is False
    assert result["status"] == "no_tests"


def test_async_reward_contract():
    sample = SimpleNamespace(
        response="def task_func(value):\n    return value + 1",
        label=_label(),
    )
    result = asyncio.run(reward(None, sample))
    assert result["acc"] is True
