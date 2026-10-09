import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "yongxiang" / "scripts" / "rejudge-code-rollout.py"
SPEC = importlib.util.spec_from_file_location("rejudge_code_rollout", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_task_summary_classification(tmp_path, monkeypatch, capsys):
    class FakeTorch:
        @staticmethod
        def load(*args, **kwargs):
            return {
                "samples": [
                    {"response": "a", "label": "x", "metadata": {"task_id": 1, "difficulty": "EASY"}},
                    {"response": "b", "label": "x", "metadata": {"task_id": 1, "difficulty": "EASY"}},
                    {"response": "c", "label": "x", "metadata": {"task_id": 2, "difficulty": "HARD"}},
                    {"response": "d", "label": "x", "metadata": {"task_id": 2, "difficulty": "HARD"}},
                ]
            }

    monkeypatch.setattr(MODULE, "torch", FakeTorch)
    monkeypatch.setattr(MODULE, "load_judge", lambda path: lambda response, label: {"acc": response == "a", "score": 1})
    summary_path = tmp_path / "summary.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        ["rejudge-code-rollout.py", "rollout.pt", "--task-summary", str(summary_path)],
    )
    MODULE.main()
    assert json.loads(summary_path.read_text().splitlines()[0])["classification"] == "mixed"
    assert json.loads(summary_path.read_text().splitlines()[1])["classification"] == "all_wrong"
    assert "all_wrong" in capsys.readouterr().out
