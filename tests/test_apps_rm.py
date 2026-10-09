import asyncio
import json
from types import SimpleNamespace

from yongxiang.apps_rm import compute_score, reward


def _label():
    return json.dumps({"task_id": "test/0", "inputs": ["2\n", "5\n"], "outputs": ["4\n", "25\n"]})


def test_apps_correct_program_passes():
    result = compute_score("```python\nprint(int(input()) ** 2)\n```", _label())
    assert result["acc"] is True
    assert result["status"] == "passed"


def test_apps_wrong_answer_is_reported():
    result = compute_score("print(int(input()))", _label())
    assert result["acc"] is False
    assert result["status"] == "wrong_answer"


def test_apps_reward_contract():
    sample = SimpleNamespace(response="print(int(input()) ** 2)", label=_label())
    assert asyncio.run(reward(None, sample))["acc"] is True
