import json
import os
import re
import subprocess
import sys
import tempfile


def _strip_reasoning_markers(response: str) -> str:
    """Remove reasoning wrappers and generation end markers from a response."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in response:
        # Chat templates may put the opening <think> marker in the prompt, so
        # the generated response can contain only the closing marker.
        response = response.split("</think>", 1)[1]
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if marker in response:
            response = response.split(marker, 1)[0]
    return response


def _extract_code(response: str, entry_point: str | None = None) -> str:
    if not isinstance(response, str):
        return ""

    response = _strip_reasoning_markers(response)

    # Match fences that occupy their own lines. This avoids treating prose such
    # as "in a ```python code block" as an opening fence.
    blocks = re.findall(
        r"(?ms)^[ \t]*```(?:python|py)?[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*(?:\r?\n|$)",
        response,
    )
    if blocks:
        if entry_point:
            function_pattern = rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\("
            matching_blocks = [block for block in blocks if re.search(function_pattern, block)]
            if matching_blocks:
                return matching_blocks[-1].strip()
        return blocks[-1].strip()

    # Some responses omit fences but still include the requested function.
    # Drop the explanation before the entry-point definition when possible.
    if entry_point:
        entry_match = re.search(
            rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", response
        )
        if entry_match:
            return response[entry_match.start() :].strip()
        # A response that never defines the requested function is reasoning or
        # other prose, not executable candidate code. Returning it here makes
        # reward logs misleading and can accidentally execute arbitrary text.
        return ""
    return response.strip()


def _run_test(code: str, entry_point: str, test_code: str, timeout: int = 5) -> bool:
    # This executes model-generated Python directly. Production deployments should
    # run it in an isolated, resource-limited, non-root sandbox without network access.
    program = code + "\n\n" + test_code
    # HumanEval-style tests expose check(candidate), while MBPP-style tests
    # execute their own input/assertion loop and must not receive a synthetic
    # check() call.
    if re.search(r"(?m)^\s*(?:async\s+)?def\s+check\s*\(", test_code):
        program += f"\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "solution.py")
        with open(path, "w", encoding="utf-8") as file:
            file.write(program)
        process = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return process.returncode == 0


def compute_score(solution_str, ground_truth, prompt=None, **kwargs) -> dict:
    del prompt, kwargs
    try:
        metadata = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        entry_point = metadata["entry_point"]
        test_code = metadata["test"]
        if not isinstance(entry_point, str) or not isinstance(test_code, str):
            raise TypeError("entry_point and test must be strings")
    except Exception:
        return {"score": -1.0, "acc": False, "pred": "[BAD_LABEL]"}

    try:
        code = _extract_code(solution_str, entry_point)
        passed = _run_test(code, entry_point, test_code)
    except Exception:
        code = solution_str if isinstance(solution_str, str) else repr(solution_str)
        passed = False

    return {
        "score": 1.0 if passed else -1.0,
        "acc": bool(passed),
        "pred": code[:200],
    }
