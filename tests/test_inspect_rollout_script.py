import subprocess
import sys
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inspect_rollout.py"


def _write_rollout(tmp_path: Path) -> Path:
    rollout_path = tmp_path / "rollout.pt"
    torch.save(
        {
            "rollout_id": 3,
            "samples": [
                {
                    "group_index": 0,
                    "index": 7,
                    "prompt": "prompt-7",
                    "response": "response-7",
                    "status": "completed",
                    "response_length": 8,
                    "reward": 1.0,
                    "metadata": {"topic": "algebra"},
                    "label": {"ground_truth": "42"},
                    "teacher_log_probs": [0.1, 0.2],
                    "rollout_log_probs": [0.3, 0.4],
                },
                {
                    "group_index": 1,
                    "index": 9,
                    "prompt": "prompt-9",
                    "response": "response-9",
                    "status": "truncated",
                    "response_length": 4,
                    "reward": 0.0,
                    "metadata": {"topic": "geometry"},
                    "label": {"ground_truth": "17"},
                    "teacher_log_probs": [0.5],
                    "rollout_log_probs": [0.6],
                },
            ],
        },
        rollout_path,
    )
    return rollout_path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        check=False,
        text=True,
        capture_output=True,
    )


def test_summary_command_reports_rollout_stats(tmp_path: Path):
    rollout_path = _write_rollout(tmp_path)

    result = _run_cli("summary", "--path", str(rollout_path))

    assert result.returncode == 0, result.stderr
    assert "rollout_id: 3" in result.stdout
    assert "num_samples: 2" in result.stdout
    assert "completed=1" in result.stdout
    assert "truncated=1" in result.stdout


def test_table_command_can_filter_truncated_samples(tmp_path: Path):
    rollout_path = _write_rollout(tmp_path)

    result = _run_cli("table", "--path", str(rollout_path), "--only-truncated")

    assert result.returncode == 0, result.stderr
    assert "index" in result.stdout
    assert "9" in result.stdout
    assert "truncated" in result.stdout
    assert "7" not in result.stdout


def test_show_command_prints_selected_sample_details(tmp_path: Path):
    rollout_path = _write_rollout(tmp_path)

    result = _run_cli("show", "--path", str(rollout_path), "--index", "7")

    assert result.returncode == 0, result.stderr
    assert "index: 7" in result.stdout
    assert "prompt-7" in result.stdout
    assert "response-7" in result.stdout
    assert "algebra" in result.stdout
    assert "teacher_log_probs" in result.stdout
