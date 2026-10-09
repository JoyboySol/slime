import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "yongxiang" / "scripts" / "convert-taco-cobalt.py"
SPEC = importlib.util.spec_from_file_location("convert_taco_cobalt", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(task_id: int, difficulty: str) -> dict:
    cases = {"inputs": [f"{task_id}\n"], "outputs": [f"{task_id}\n"]}
    encoded = json.dumps(cases)
    return {
        "id": task_id,
        "difficulty": difficulty,
        "instruction": f"Solve task {task_id}",
        "test_cases": encoded,
        "public_test_cases": encoded,
        "hidden_test_cases": encoded,
    }


def test_convert_samples_train_rows_deterministically(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = [_row(task_id, "EASY" if task_id % 2 else "HARD") for task_id in range(10)]
    (source / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    ids = MODULE.convert(source, first, "train", 3, 42, "hidden", {"HARD"})
    assert ids == MODULE.convert(source, second, "train", 3, 42, "hidden", {"HARD"})
    assert first.read_bytes() == second.read_bytes()

    records = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert all(record["metadata"]["split"] == "train" for record in records)
    assert all(record["metadata"]["difficulty"] == "HARD" for record in records)
    assert all(record["metadata"]["test_split"] == "hidden" for record in records)
