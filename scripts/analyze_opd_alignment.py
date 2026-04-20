#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import pandas as pd

from slime.utils.opd_utils import (
    align_token_byte_chunks,
    build_contextual_suffix_token_bytes,
    build_token_byte_spans,
    clip_token_bytes_by_region,
    decode_token_ids,
    encode_text,
    get_cached_tokenizer,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass
class AnalysisResult:
    sample_id: str
    source_file: str
    row_index: int
    status: str
    error: str | None
    prompt_token_count: int
    response_token_count: int
    teacher_response_token_count: int
    aligned_chunk_count: int | None
    one_to_one_aligned_token_count: int | None
    student_response_bytes: int
    teacher_response_bytes: int
    student_reconstruction_mode: str
    render_full_matches_decoded: bool
    render_prompt_matches_decoded: bool
    first_diff_byte: int | None
    first_diff_student_preview: str | None
    first_diff_teacher_preview: str | None
    response_preview: str


def discover_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    files = sorted(file for file in path.rglob("*") if file.suffix in {".parquet", ".jsonl"})
    if not files:
        raise FileNotFoundError(f"No .parquet or .jsonl files found under: {path}")
    return files


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _normalize_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_messages = row.get("messages")
    if raw_messages is None:
        messages = []
    else:
        messages = [_jsonable(message) for message in list(raw_messages)]

    system = row.get("system")
    if system and (not messages or messages[0].get("role") != "system"):
        messages = [{"role": "system", "content": system}] + messages

    if not messages:
        raise ValueError("Row does not contain any messages.")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Last message is not assistant; cannot derive response boundary.")
    return messages


def iter_rows(path: Path) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for file_path in discover_input_files(path):
        if file_path.suffix == ".parquet":
            frame = pd.read_parquet(file_path)
            for row_index, row in enumerate(frame.to_dict(orient="records")):
                yield file_path, row_index, row
        elif file_path.suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as handle:
                for row_index, line in enumerate(handle):
                    if line.strip():
                        yield file_path, row_index, json.loads(line)


def _apply_chat_template(tokenizer, messages: list[dict[str, Any]], *, tokenize: bool, tools: Any = None):
    kwargs = {
        "tokenize": tokenize,
        "return_dict": False,
    }
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def _apply_prompt_template(tokenizer, messages: list[dict[str, Any]], *, tokenize: bool, tools: Any = None):
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": True,
        "return_dict": False,
    }
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def _compute_first_diff(student_bytes: bytes, teacher_bytes: bytes) -> tuple[int | None, str | None, str | None]:
    for index, (student_byte, teacher_byte) in enumerate(zip(student_bytes, teacher_bytes, strict=False)):
        if student_byte != teacher_byte:
            return (
                index,
                repr(student_bytes[max(0, index - 12) : index + 12].decode("utf-8", errors="replace")),
                repr(teacher_bytes[max(0, index - 12) : index + 12].decode("utf-8", errors="replace")),
            )
    if len(student_bytes) != len(teacher_bytes):
        index = min(len(student_bytes), len(teacher_bytes))
        return (
            index,
            repr(student_bytes[max(0, index - 12) : index + 12].decode("utf-8", errors="replace")),
            repr(teacher_bytes[max(0, index - 12) : index + 12].decode("utf-8", errors="replace")),
        )
    return None, None, None


def analyze_row(
    row: dict[str, Any],
    *,
    source_file: Path,
    row_index: int,
    student_tokenizer,
    teacher_tokenizer,
) -> AnalysisResult:
    messages = _normalize_messages(row)
    raw_tools = row.get("tools")
    tools = _jsonable(raw_tools) if raw_tools is not None else None
    sample_id = str(row.get("sample_id") or f"{source_file}:{row_index}")

    prompt_messages = messages[:-1]
    if not prompt_messages:
        raise ValueError("Need at least one non-assistant message before final assistant response.")

    rendered_full_text = _apply_chat_template(student_tokenizer, messages, tokenize=False, tools=tools)
    rendered_prompt_text = _apply_prompt_template(student_tokenizer, prompt_messages, tokenize=False, tools=tools)

    student_full_token_ids = list(_apply_chat_template(student_tokenizer, messages, tokenize=True, tools=tools))
    prompt_token_ids = list(_apply_prompt_template(student_tokenizer, prompt_messages, tokenize=True, tools=tools))
    if len(student_full_token_ids) < len(prompt_token_ids):
        raise ValueError("Prompt token count is larger than full token count.")

    response_token_ids = student_full_token_ids[len(prompt_token_ids) :]
    decoded_full_text = decode_token_ids(student_tokenizer, student_full_token_ids)
    decoded_prompt_text = decode_token_ids(student_tokenizer, prompt_token_ids)
    decoded_response_text = decode_token_ids(student_tokenizer, response_token_ids)

    student_response_token_bytes, _student_response_spans = build_contextual_suffix_token_bytes(
        student_tokenizer,
        student_full_token_ids,
        len(prompt_token_ids),
        rendered_prompt_text,
        full_text=rendered_full_text,
    )
    student_reconstruction_mode = "contextual_suffix"

    teacher_token_ids = encode_text(teacher_tokenizer, rendered_full_text)
    teacher_token_bytes, teacher_token_spans = build_token_byte_spans(teacher_tokenizer, rendered_full_text, teacher_token_ids)
    teacher_response_token_bytes, _teacher_response_indices = clip_token_bytes_by_region(
        teacher_token_bytes,
        teacher_token_spans,
        start_byte=len(rendered_prompt_text.encode("utf-8")),
    )

    student_response_bytes = b"".join(student_response_token_bytes)
    teacher_response_bytes = b"".join(teacher_response_token_bytes)
    first_diff_byte, first_diff_student_preview, first_diff_teacher_preview = _compute_first_diff(
        student_response_bytes, teacher_response_bytes
    )

    status = "ok"
    error = None
    aligned_chunk_count = None
    one_to_one_aligned_token_count = None
    try:
        aligned_chunks = align_token_byte_chunks(student_response_token_bytes, teacher_response_token_bytes)
        aligned_chunk_count = len(aligned_chunks)
        one_to_one_aligned_token_count = sum(
            1
            for student_slice, teacher_slice, _chunk_bytes in aligned_chunks
            if (student_slice.stop - student_slice.start) == 1 and (teacher_slice.stop - teacher_slice.start) == 1
        )
    except ValueError as exc:
        error = str(exc)
        status = "response_bytes_mismatch" if student_response_bytes != teacher_response_bytes else "chunk_alignment_failed"

    return AnalysisResult(
        sample_id=sample_id,
        source_file=str(source_file),
        row_index=row_index,
        status=status,
        error=error,
        prompt_token_count=len(prompt_token_ids),
        response_token_count=len(response_token_ids),
        teacher_response_token_count=len(teacher_response_token_bytes),
        aligned_chunk_count=aligned_chunk_count,
        one_to_one_aligned_token_count=one_to_one_aligned_token_count,
        student_response_bytes=len(student_response_bytes),
        teacher_response_bytes=len(teacher_response_bytes),
        student_reconstruction_mode=student_reconstruction_mode,
        render_full_matches_decoded=(rendered_full_text == decoded_full_text),
        render_prompt_matches_decoded=(rendered_prompt_text == decoded_prompt_text),
        first_diff_byte=first_diff_byte,
        first_diff_student_preview=first_diff_student_preview,
        first_diff_teacher_preview=first_diff_teacher_preview,
        response_preview=repr(decoded_response_text[:160]),
    )


def analyze_path(
    path: Path,
    *,
    student_tokenizer_path: str,
    teacher_tokenizer_path: str,
    limit: int | None,
) -> list[AnalysisResult]:
    student_tokenizer = get_cached_tokenizer(student_tokenizer_path)
    teacher_tokenizer = get_cached_tokenizer(teacher_tokenizer_path)

    results: list[AnalysisResult] = []
    for file_path, row_index, row in iter_rows(path):
        if limit is not None and len(results) >= limit:
            break
        sample_id = str(row.get("sample_id") or f"{file_path}:{row_index}")
        try:
            result = analyze_row(
                row,
                source_file=file_path,
                row_index=row_index,
                student_tokenizer=student_tokenizer,
                teacher_tokenizer=teacher_tokenizer,
            )
        except Exception as exc:  # pragma: no cover - surfaced by CLI
            result = AnalysisResult(
                sample_id=sample_id,
                source_file=str(file_path),
                row_index=row_index,
                status="sample_error",
                error=str(exc),
                prompt_token_count=0,
                response_token_count=0,
                teacher_response_token_count=0,
                aligned_chunk_count=None,
                one_to_one_aligned_token_count=None,
                student_response_bytes=0,
                teacher_response_bytes=0,
                student_reconstruction_mode="n/a",
                render_full_matches_decoded=False,
                render_prompt_matches_decoded=False,
                first_diff_byte=None,
                first_diff_student_preview=None,
                first_diff_teacher_preview=None,
                response_preview="",
            )
        results.append(result)
    return results


def results_to_frame(results: list[AnalysisResult]) -> pd.DataFrame:
    frame = pd.DataFrame(asdict(result) for result in results)
    frame = frame.rename(columns={"response_token_count": "student_response_token_count"})

    numeric_columns = [
        "prompt_token_count",
        "student_response_token_count",
        "teacher_response_token_count",
        "aligned_chunk_count",
        "one_to_one_aligned_token_count",
        "student_response_bytes",
        "teacher_response_bytes",
        "first_diff_byte",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    chunk_count = frame["aligned_chunk_count"].replace(0, pd.NA)
    student_token_count = frame["student_response_token_count"].replace(0, pd.NA)
    teacher_token_count = frame["teacher_response_token_count"].replace(0, pd.NA)

    frame["student_tokens_per_chunk"] = frame["student_response_token_count"] / chunk_count
    frame["teacher_tokens_per_chunk"] = frame["teacher_response_token_count"] / chunk_count
    frame["teacher_to_student_token_ratio"] = teacher_token_count / student_token_count
    frame["chunk_to_student_token_ratio"] = frame["aligned_chunk_count"] / student_token_count
    frame["student_one_to_one_aligned_token_ratio"] = frame["one_to_one_aligned_token_count"] / student_token_count
    frame["teacher_one_to_one_aligned_token_ratio"] = frame["one_to_one_aligned_token_count"] / teacher_token_count
    frame["byte_length_delta"] = frame["teacher_response_bytes"] - frame["student_response_bytes"]
    return frame


def _metric_rows(series: pd.Series, prefix: str) -> list[dict[str, float]]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return [{"metric": f"{prefix}_{name}", "value": float("nan")} for name in ("mean", "median", "p05", "p95", "sum")]
    return [
        {"metric": f"{prefix}_mean", "value": float(clean.mean())},
        {"metric": f"{prefix}_median", "value": float(clean.median())},
        {"metric": f"{prefix}_p05", "value": float(clean.quantile(0.05))},
        {"metric": f"{prefix}_p95", "value": float(clean.quantile(0.95))},
        {"metric": f"{prefix}_sum", "value": float(clean.sum())},
    ]


def build_report_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok_frame = frame[frame["status"] == "ok"].copy()
    summary_rows: list[dict[str, float]] = [
        {"metric": "sample_count", "value": float(len(frame))},
        {"metric": "ok_sample_count", "value": float(len(ok_frame))},
        {"metric": "ok_sample_ratio", "value": float(len(ok_frame) / len(frame)) if len(frame) else float("nan")},
    ]
    for column in (
        "student_response_token_count",
        "teacher_response_token_count",
        "aligned_chunk_count",
        "one_to_one_aligned_token_count",
        "student_tokens_per_chunk",
        "teacher_tokens_per_chunk",
        "teacher_to_student_token_ratio",
        "chunk_to_student_token_ratio",
        "student_one_to_one_aligned_token_ratio",
        "teacher_one_to_one_aligned_token_ratio",
        "student_response_bytes",
        "teacher_response_bytes",
    ):
        if column in ok_frame:
            summary_rows.extend(_metric_rows(ok_frame[column], column))

    summary_frame = pd.DataFrame(summary_rows)
    status_frame = (
        frame.groupby("status", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "status"], ascending=[False, True], ignore_index=True)
    )
    return summary_frame, status_frame


def _save_token_count_overview(ok_frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plots = [
        ("student_response_token_count", "Student response tokens", "#1f77b4"),
        ("teacher_response_token_count", "Teacher response tokens", "#ff7f0e"),
        ("aligned_chunk_count", "Aligned chunks", "#2ca02c"),
    ]
    for ax, (column, title, color) in zip(axes, plots, strict=False):
        values = ok_frame[column].dropna()
        bins = min(40, max(10, len(values))) if len(values) else 10
        ax.hist(values, bins=bins, color=color, alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.set_ylabel("Samples")
        ax.grid(alpha=0.2)
    fig.suptitle("OPD token and chunk count overview")
    fig.tight_layout()
    fig.savefig(output_dir / "token_count_overview.png", dpi=180)
    plt.close(fig)


def _save_alignment_scatter(ok_frame: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        ok_frame["student_response_token_count"],
        ok_frame["teacher_response_token_count"],
        c=ok_frame["aligned_chunk_count"],
        cmap="viridis",
        alpha=0.8,
        s=28,
    )
    if not ok_frame.empty:
        max_value = max(
            ok_frame["student_response_token_count"].max(),
            ok_frame["teacher_response_token_count"].max(),
        )
        ax.plot([0, max_value], [0, max_value], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Teacher vs student response token counts")
    ax.set_xlabel("Student response tokens")
    ax.set_ylabel("Teacher response tokens")
    ax.grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Aligned chunk count")
    fig.tight_layout()
    fig.savefig(output_dir / "token_alignment_scatter.png", dpi=180)
    plt.close(fig)


def _save_chunk_ratio_distributions(ok_frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plots = [
        ("student_tokens_per_chunk", "Student tokens per chunk", "#1f77b4"),
        ("teacher_tokens_per_chunk", "Teacher tokens per chunk", "#ff7f0e"),
        ("teacher_to_student_token_ratio", "Teacher / student token ratio", "#d62728"),
        ("chunk_to_student_token_ratio", "Aligned chunks / student tokens", "#2ca02c"),
        ("student_one_to_one_aligned_token_ratio", "1:1 aligned / student tokens", "#9467bd"),
        ("teacher_one_to_one_aligned_token_ratio", "1:1 aligned / teacher tokens", "#8c564b"),
    ]
    for ax, (column, title, color) in zip(axes.flatten(), plots, strict=False):
        values = ok_frame[column].dropna()
        bins = min(40, max(10, len(values))) if len(values) else 10
        ax.hist(values, bins=bins, color=color, alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Samples")
        ax.grid(alpha=0.2)
    fig.suptitle("Chunk density and tokenizer expansion ratios")
    fig.tight_layout()
    fig.savefig(output_dir / "chunk_ratio_distributions.png", dpi=180)
    plt.close(fig)


def write_report(frame: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame, status_frame = build_report_tables(frame)
    frame.to_csv(output_dir / "sample_metrics.csv", index=False)
    summary_frame.to_csv(output_dir / "summary_metrics.csv", index=False)
    status_frame.to_csv(output_dir / "status_counts.csv", index=False)

    ok_frame = frame[frame["status"] == "ok"].copy()
    _save_token_count_overview(ok_frame, output_dir)
    _save_alignment_scatter(ok_frame, output_dir)
    _save_chunk_ratio_distributions(ok_frame, output_dir)
    return summary_frame, status_frame


def print_summary(results: list[AnalysisResult]) -> None:
    frame = results_to_frame(results)
    summary_frame, status_frame = build_report_tables(frame)
    status_counts = Counter(result.status for result in results)
    reconstruction_counts = Counter(result.student_reconstruction_mode for result in results)
    render_full_mismatch_count = sum(not result.render_full_matches_decoded for result in results)
    render_prompt_mismatch_count = sum(not result.render_prompt_matches_decoded for result in results)

    print(f"num_samples: {len(results)}")
    print("status_counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print("student_reconstruction_mode:")
    for mode, count in sorted(reconstruction_counts.items()):
        print(f"  {mode}: {count}")
    print(f"render_full_mismatch_count: {render_full_mismatch_count}")
    print(f"render_prompt_mismatch_count: {render_prompt_mismatch_count}")
    print("ok_sample_metrics:")
    for row in summary_frame.itertuples(index=False):
        if not row.metric.startswith("sample_count") and not row.metric.startswith("ok_sample"):
            print(f"  {row.metric}: {row.value:.6f}")

    failures = [result for result in results if result.status != "ok"]
    if failures:
        print("top_failures:")
        for result in failures[:10]:
            print(
                "  "
                + json.dumps(
                    {
                        "sample_id": result.sample_id,
                        "status": result.status,
                        "error": result.error,
                        "first_diff_byte": result.first_diff_byte,
                        "student_preview": result.first_diff_student_preview,
                        "teacher_preview": result.first_diff_teacher_preview,
                    },
                    ensure_ascii=False,
                )
            )
    print("status_table:")
    for row in status_frame.itertuples(index=False):
        print(f"  {row.status}: {row.count}")


def print_failures(results: list[AnalysisResult], *, limit: int) -> None:
    failures = [result for result in results if result.status != "ok"]
    for result in failures[:limit]:
        print(json.dumps(asdict(result), ensure_ascii=False))


def print_show(results: list[AnalysisResult], *, sample_id: str) -> None:
    for result in results:
        if result.sample_id == sample_id:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
            return
    raise ValueError(f"sample_id not found in analyzed results: {sample_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze cross-tokenizer OPD byte_chunk alignment on sample shards.")
    parser.add_argument("--path", default='/mnt/hdd/lvzhihao/output/OpenMathInstruct-2/correct/shards', help="Path to a parquet/jsonl file or a directory of sample shards.")
    parser.add_argument("--student-tokenizer", default='/mnt/hdd/lvzhihao/hf_models/Dist-mathcode10b-s1randg-sch1-CPT-200b-stage3-r640k-GDN2.9b-A7-12_20_21_23_46_48_49-sl32768bs128lr2e5-2e5/merged_10ckpts_iter_61984-hf_to_iter_71525-hf_mean', help="HF checkpoint/path for the student tokenizer.")
    parser.add_argument("--teacher-tokenizer", default='/mnt/hdd/Nanbeige4.1-3B', help="HF checkpoint/path for the teacher tokenizer.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of samples to analyze.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("summary", help="Print aggregated failure counts and representative failures.")

    report_parser = subparsers.add_parser("report", help="Write per-sample metrics, summary tables, and plots.")
    report_parser.add_argument(
        "--output-dir",
        default=str(Path(".tmp") / "analyze_opd_alignment_report"),
        help="Directory to write CSV summaries and PNG plots.",
    )

    failures_parser = subparsers.add_parser("failures", help="Print one JSON record per failed sample.")
    failures_parser.add_argument("--fail-limit", type=int, default=20, help="Maximum failed samples to print.")

    show_parser = subparsers.add_parser("show", help="Print one analyzed sample by sample_id.")
    show_parser.add_argument("--sample-id", required=True, help="Sample ID to display.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        results = analyze_path(
            Path(args.path),
            student_tokenizer_path=args.student_tokenizer,
            teacher_tokenizer_path=args.teacher_tokenizer,
            limit=args.limit,
        )
        if args.command == "summary":
            print_summary(results)
        elif args.command == "report":
            frame = results_to_frame(results)
            summary_frame, status_frame = write_report(frame, Path(args.output_dir))
            print(f"report_dir: {Path(args.output_dir).resolve()}")
            print("written_files:")
            for name in (
                "sample_metrics.csv",
                "summary_metrics.csv",
                "status_counts.csv",
                "token_count_overview.png",
                "token_alignment_scatter.png",
                "chunk_ratio_distributions.png",
            ):
                print(f"  {name}")
            print("headline_metrics:")
            metrics = dict(zip(summary_frame["metric"], summary_frame["value"], strict=False))
            for key in (
                "sample_count",
                "ok_sample_count",
                "student_response_token_count_mean",
                "teacher_response_token_count_mean",
                "aligned_chunk_count_mean",
                "teacher_to_student_token_ratio_mean",
                "chunk_to_student_token_ratio_mean",
            ):
                print(f"  {key}: {metrics.get(key)}")
            print("status_counts:")
            for row in status_frame.itertuples(index=False):
                print(f"  {row.status}: {row.count}")
        elif args.command == "failures":
            print_failures(results, limit=args.fail_limit)
        elif args.command == "show":
            print_show(results, sample_id=args.sample_id)
        else:  # pragma: no cover
            parser.error(f"Unsupported command: {args.command}")
    except Exception as exc:  # pragma: no cover - surfaced through CLI
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
