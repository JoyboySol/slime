from __future__ import annotations

from collections import Counter
from typing import Any


def _normalize_metric_fragment(value: Any) -> str:
    text = "none" if value is None else str(value)
    normalized = []
    for char in text.lower():
        if char.isalnum():
            normalized.append(char)
        else:
            normalized.append("_")
    collapsed = "".join(normalized).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed or "empty"


def summarize_opd_alignment(
    *,
    status_values: list[Any] | None,
    source_values: list[Any] | None,
    validated_values: list[Any] | None,
    error_values: list[Any] | None,
) -> tuple[dict[str, float], str | None]:
    status_counter = Counter(status_values or [])
    source_counter = Counter(source_values or [])
    validated_counter = Counter(validated_values or [])
    error_counter = Counter(error for error in (error_values or []) if error)

    sample_count = max(
        len(status_values or []),
        len(source_values or []),
        len(validated_values or []),
        len(error_values or []),
    )
    if sample_count == 0 and not status_counter and not source_counter and not validated_counter and not error_counter:
        return {}, None

    metrics: dict[str, float] = {
        "opd_alignment_sample_count": float(sample_count),
        "opd_alignment_error_count": float(sum(error_counter.values())),
        "opd_alignment_unique_error_count": float(len(error_counter)),
    }
    for status, count in status_counter.items():
        metrics[f"opd_alignment_status_{_normalize_metric_fragment(status)}_count"] = float(count)
    for source, count in source_counter.items():
        metrics[f"opd_alignment_source_{_normalize_metric_fragment(source)}_count"] = float(count)
    for validated, count in validated_counter.items():
        metrics[f"opd_alignment_validated_{_normalize_metric_fragment(validated)}_count"] = float(count)

    summary_text = (
        f"status={dict(status_counter)} "
        f"source={dict(source_counter)} "
        f"validated={dict(validated_counter)} "
        f"top_errors={error_counter.most_common(3)}"
    )
    return metrics, summary_text


def summarize_opd_alignment_from_samples(samples: list[Any]) -> tuple[dict[str, float], str | None]:
    return summarize_opd_alignment(
        status_values=[getattr(sample, "opd_student_alignment_status", None) for sample in samples],
        source_values=[getattr(sample, "opd_student_alignment_source", None) for sample in samples],
        validated_values=[getattr(sample, "opd_student_alignment_validated", None) for sample in samples],
        error_values=[getattr(sample, "opd_student_alignment_error", None) for sample in samples],
    )


def summarize_opd_alignment_from_rollout_data(rollout_data: dict[str, Any]) -> tuple[dict[str, float], str | None]:
    return summarize_opd_alignment(
        status_values=rollout_data.get("opd_student_alignment_status_list"),
        source_values=rollout_data.get("opd_student_alignment_source_list"),
        validated_values=rollout_data.get("opd_student_alignment_validated_list"),
        error_values=rollout_data.get("opd_student_alignment_error_list"),
    )
