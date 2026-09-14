"""Create a conservative math-training dataset with judgeable answers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from math_verify import parse


BOXED_RE = re.compile(r"\\boxed\{(.+)\}", re.DOTALL)
DISPLAY_RE = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)|\$\$(.+?)\$\$", re.DOTALL)
INLINE_RE = re.compile(r"\$(?!\$)(.+?)(?<!\$)\$", re.DOTALL)
BARE_LATEX_RE = re.compile(
    r"\\(?:begin|end|frac|dfrac|tfrac|sqrt|pi|pm|circ|mathbf|mathbb|left|right|"
    r"text|mathrm|overline|vec|underline|cases|array)\b"
)
BARE_OPERATOR_RE = re.compile(r"(?=.*\d)(?=.*[+\-*/^=])^[0-9A-Za-z_+\-*/^=().,:\\{}\s]+$")


def _balanced_boxed(text: str) -> str | None:
    idx = text.rfind(r"\boxed{")
    if idx < 0:
        return None
    depth = 0
    for pos in range(idx + len(r"\boxed{"), len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            if depth == 0:
                return text[idx + len(r"\boxed{") : pos].strip()
            depth -= 1
    return None


def extract_answer(raw: str) -> tuple[str | None, str]:
    """Extract one answer, rejecting prose with no unambiguous math span."""
    text = str(raw).strip()
    if re.fullmatch(r"[A-Da-d]", text):
        return text.upper(), "choice"
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return text, "numeric"

    boxed = _balanced_boxed(text)
    if boxed:
        return boxed, "boxed"

    spans = [m.group(1) or m.group(2) or m.group(3) for m in DISPLAY_RE.finditer(text)]
    spans += [m.group(1) for m in INLINE_RE.finditer(text)]
    spans = [s.strip() for s in spans if s and s.strip()]
    if spans:
        return spans[-1], "delimited_math"

    text = text.strip().strip(".$")
    if len(text) > 200 or "\n" in text:
        return None, "long_or_multiline_prose"
    if not BARE_LATEX_RE.search(text) and not BARE_OPERATOR_RE.search(text):
        return None, "no_unambiguous_math_marker"
    # Reject ordinary explanatory prose while allowing variables such as x, n,
    # and LaTeX command names beginning with a backslash.
    word_check = re.sub(r"\\(?:begin|end)\{[^}]+\}", "", text)
    word_check = re.sub(r"\\[A-Za-z]+", "", word_check)
    words = re.findall(r"(?<![A-Za-z])[A-Za-z]{3,}(?![A-Za-z])", word_check)
    if words:
        return None, "contains_prose"
    return text, "bare_latex"


def parseable(answer: str) -> bool:
    variants = [answer, f"${answer}$", rf"\boxed{{{answer}}}"]
    for variant in variants:
        try:
            if parse(variant, extraction_mode="any_match"):
                return True
        except Exception:
            continue
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "aime_like_judgeable.jsonl"
    dropped = args.output_dir / "dropped_labels.jsonl"

    kept = dropped_count = 0
    reasons: dict[str, int] = {}
    with args.input.open(encoding="utf-8") as src, output.open("w", encoding="utf-8") as out, dropped.open(
        "w", encoding="utf-8"
    ) as bad:
        for line_number, line in enumerate(src, start=1):
            record = json.loads(line)
            original = str(record.get("answer", ""))
            answer, reason = extract_answer(original)
            if answer is None or not parseable(answer):
                record["filter_reason"] = reason if answer is None else "math_verify_unparseable"
                bad.write(json.dumps(record, ensure_ascii=False) + "\n")
                dropped_count += 1
                reasons[record["filter_reason"]] = reasons.get(record["filter_reason"], 0) + 1
                continue
            record["original_answer"] = original
            record["answer"] = answer
            record["judge_answer_kind"] = reason
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    manifest = {
        "input": str(args.input),
        "output": str(output),
        "dropped": str(dropped),
        "kept": kept,
        "dropped_count": dropped_count,
        "drop_reasons": reasons,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
