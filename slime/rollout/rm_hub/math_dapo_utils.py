# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re

try:
    from math_verify import parse as math_verify_parse
    from math_verify import verify as math_verify_verify
except ImportError:  # pragma: no cover - optional dependency in minimal installs
    math_verify_parse = None
    math_verify_verify = None
import signal


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


class timeout:

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
    "<|im_end|>",
    "<|endoftext|>",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = str(final_answer)
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Prefer the final boxed expression when available.  This also works for
    # long chain-of-thought responses where the answer is not written as
    # ``Answer: ...``.
    boxed = last_boxed_only_string(solution_str)
    if boxed is not None:
        extracted_answer = remove_boxed(boxed)
    else:
        match = re.findall(answer_pattern, solution_str)
        extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract and (boxed_gt := last_boxed_only_string(gt)) is not None:
        gt = normalize_final_answer(remove_boxed(boxed_gt))
    else:
        gt = normalize_final_answer(gt)

    # DAPO originally assumed integer-only answers.  The training corpus also
    # contains letter and symbolic answers, so compare numeric values when
    # both sides are numeric and otherwise compare normalized strings.
    numeric_pattern = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$"
    if re.fullmatch(numeric_pattern, pred.replace(",", "")) and re.fullmatch(numeric_pattern, gt.replace(",", "")):
        pred_value = float(pred.replace(",", ""))
        gt_value = float(gt.replace(",", ""))
        # Keep integer answers readable while still treating ``42`` and
        # ``42.0`` as the same numeric value.
        pred = str(int(pred_value)) if pred_value.is_integer() else str(pred_value)
        gt = str(int(gt_value)) if gt_value.is_integer() else str(gt_value)
    else:
        pred = pred.strip().casefold()
        gt = gt.strip().casefold()

    return (pred == gt), pred


def is_correct_strict_box(pred: str, gt: str, pause_tokens_index: list[int] | None = None) -> tuple[int, str | None]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def _math_verify_variants(text: str) -> list[str]:
    """Return parseable variants for a math expression.

    ``math_verify`` treats a bare LaTeX fragment such as ``\\begin{pmatrix}...
    \\end{pmatrix}`` as prose and may extract only its last scalar.  Dataset
    labels frequently contain exactly these bare fragments, while model
    answers usually put them inside ``$...$`` or ``\\boxed{...}``.  Add math
    delimiters only for delimiter-free, math-looking text and keep the raw
    form first for prose answers.
    """
    text = str(text).strip()
    variants = [text]
    has_delimiter = any(token in text for token in ("$", r"\(", r"\[", r"\boxed{"))
    looks_like_math = bool(
        re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text)
        or re.search(r"\\(?:begin|end|frac|dfrac|tfrac|sqrt|left|right|text|mathrm)\b", text)
    )
    if text and not has_delimiter and looks_like_math:
        variants.extend([f"${text}$", rf"\boxed{{{text}}}"])
    return variants


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: list[int] | None = None
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    # Prefer math-verify when available.  Unlike the original DAPO checker,
    # this handles mathematically equivalent fractions, decimals, radicals,
    # equations, sets, and other non-integer answers.  Prefer the last boxed
    # answer when present: parsing an entire long chain of thought can select
    # an intermediate number or fail on an incomplete expression near the
    # truncation boundary.
    if math_verify_parse is not None and math_verify_verify is not None:
        candidates = []
        boxed = last_boxed_only_string(solution_str)
        if boxed is not None:
            candidates.append(remove_boxed(boxed))
        candidates.append(solution_str[-2000:])
        try:
            for expected_text in _math_verify_variants(answer):
                try:
                    expected = math_verify_parse(expected_text, extraction_mode="any_match")
                except Exception:
                    continue
                if not expected:
                    continue
                for candidate in candidates:
                    for candidate_text in _math_verify_variants(candidate):
                        try:
                            predicted = math_verify_parse(candidate_text, extraction_mode="any_match")
                            if predicted and math_verify_verify(expected, predicted, strict=True):
                                return True, predicted[-1]
                        except Exception:
                            continue
        except Exception:
            pass

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: list[int] | None = None,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Verify the solution
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }
