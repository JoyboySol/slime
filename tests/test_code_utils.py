from slime.rollout.rm_hub.code_utils import _extract_code, compute_score


NUM_GPUS = 0


def test_extract_code_ignores_inline_fence_mentions():
    response = """Explain that the answer belongs in a ```python code block.

```python
def solve(value):
    return value + 1
```
"""
    assert _extract_code(response, "solve") == "def solve(value):\n    return value + 1"


def test_extract_code_prefers_block_with_entry_point():
    response = """```python
def unrelated(value):
    return value
```

```python
def solve(value):
    return value + 1
```
<|im_end|>"""
    assert "def solve(value):" in _extract_code(response, "solve")


def test_extract_code_removes_reasoning_markers():
    response = """<think>reasoning that should not be executed</think>
```python
def solve(value):
    return value + 1
```
<|im_end|>"""
    assert _extract_code(response, "solve") == "def solve(value):\n    return value + 1"


def test_extract_code_handles_prompt_reasoning_prefix():
    response = """Reasoning from the assistant\n</think>\n\n```python\ndef solve(value):\n    return value + 1\n```"""
    assert _extract_code(response, "solve") == "def solve(value):\n    return value + 1"


def test_extract_code_recovers_unfenced_entry_point():
    response = """The implementation is:

def solve(value):
    return value + 1
"""
    assert _extract_code(response, "solve").startswith("def solve(value):")


def test_extract_code_rejects_reasoning_without_entry_point():
    response = "We need to reason about the implementation before writing code."
    assert _extract_code(response, "solve") == ""


def test_compute_score_runs_tests_without_check_wrapper():
    label = {
        "entry_point": "solve",
        "test": "assert solve(1) == 2\nassert solve(3) == 4",
    }
    result = compute_score("```python\ndef solve(value):\n    return value + 1\n```", label)
    assert result["acc"] is True


def test_compute_score_keeps_humaneval_check_wrapper():
    label = {
        "entry_point": "solve",
        "test": "def check(candidate):\n    assert candidate(1) == 2",
    }
    result = compute_score("def solve(value):\n    return value + 1", label)
    assert result["acc"] is True


def test_compute_score_handles_end_marker_after_fence():
    label = {
        "entry_point": "solve",
        "test": "assert solve(1) == 2",
    }
    response = "```python\ndef solve(value):\n    return value + 1\n```<|im_end|>"
    assert compute_score(response, label)["acc"] is True


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
