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


def _compute_generation_byte_evidence_metrics(
    source_values: list[Any] | None,
    complete_values: list[Any] | None,
    validated_values: list[Any] | None,
    generation_attempted_values: list[Any] | None = None,
    generation_complete_values: list[Any] | None = None,
    generation_validated_values: list[Any] | None = None,
) -> dict[str, float]:
    source_list = list(source_values or [])
    complete_list = list(complete_values or [])
    validated_list = list(validated_values or [])
    generation_attempted_list = list(generation_attempted_values or [])
    generation_complete_list = list(generation_complete_values or [])
    generation_validated_list = list(generation_validated_values or [])
    sample_count = max(
        len(source_list),
        len(complete_list),
        len(validated_list),
        len(generation_attempted_list),
        len(generation_complete_list),
        len(generation_validated_list),
    )
    if sample_count == 0:
        return {}

    if any(value is not None for value in generation_attempted_list):
        hit_indices = [index for index, attempted in enumerate(generation_attempted_list) if attempted is True]
        complete_metric_list = generation_complete_list
        validated_metric_list = generation_validated_list
    else:
        hit_indices = [index for index, source in enumerate(source_list) if source == "generation_byte_evidence"]
        complete_metric_list = complete_list
        validated_metric_list = validated_list
    hit_count = len(hit_indices)

    incomplete_count = 0
    invalid_count = 0
    for index in hit_indices:
        complete = complete_metric_list[index] if index < len(complete_metric_list) else None
        validated = validated_metric_list[index] if index < len(validated_metric_list) else None
        if complete is False:
            incomplete_count += 1
        if validated is False:
            invalid_count += 1

    return {
        "opd_generation_byte_evidence_sample_count": float(hit_count),
        "opd_generation_byte_evidence_hit_count": float(hit_count),
        "opd_generation_byte_evidence_hit_rate": float(hit_count / sample_count),
        "opd_generation_byte_evidence_incomplete_count": float(incomplete_count),
        "opd_generation_byte_evidence_incomplete_rate": float(incomplete_count / hit_count) if hit_count else 0.0,
        "opd_generation_byte_evidence_invalid_count": float(invalid_count),
        "opd_generation_byte_evidence_invalid_rate": float(invalid_count / hit_count) if hit_count else 0.0,
    }


def summarize_opd_alignment(
    *,
    status_values: list[Any] | None,
    complete_values: list[Any] | None,
    source_values: list[Any] | None,
    validated_values: list[Any] | None,
    error_values: list[Any] | None,
    metadata_values: list[Any] | None = None,
    generation_attempted_values: list[Any] | None = None,
    generation_complete_values: list[Any] | None = None,
    generation_validated_values: list[Any] | None = None,
) -> tuple[dict[str, float], str | None]:
    status_counter = Counter(status_values or [])
    complete_counter = Counter(complete_values or [])
    source_counter = Counter(source_values or [])
    validated_counter = Counter(validated_values or [])
    evidence_kind_counter = Counter(
        metadata.get("evidence_kind")
        for metadata in (metadata_values or [])
        if isinstance(metadata, dict) and metadata.get("evidence_kind") is not None
    )
    error_counter = Counter(error for error in (error_values or []) if error)

    sample_count = max(
        len(status_values or []),
        len(complete_values or []),
        len(source_values or []),
        len(validated_values or []),
        len(error_values or []),
    )
    if sample_count == 0 and not status_counter and not complete_counter and not source_counter and not validated_counter and not error_counter:
        return {}, None

    metrics: dict[str, float] = {
        "opd_alignment_sample_count": float(sample_count),
        "opd_alignment_error_count": float(sum(error_counter.values())),
        "opd_alignment_unique_error_count": float(len(error_counter)),
    }
    for status, count in status_counter.items():
        metrics[f"opd_alignment_status_{_normalize_metric_fragment(status)}_count"] = float(count)
    for complete, count in complete_counter.items():
        metrics[f"opd_alignment_complete_{_normalize_metric_fragment(complete)}_count"] = float(count)
    for source, count in source_counter.items():
        metrics[f"opd_alignment_source_{_normalize_metric_fragment(source)}_count"] = float(count)
    for validated, count in validated_counter.items():
        metrics[f"opd_alignment_validated_{_normalize_metric_fragment(validated)}_count"] = float(count)
    for evidence_kind, count in evidence_kind_counter.items():
        metrics[f"opd_alignment_evidence_kind_{_normalize_metric_fragment(evidence_kind)}_count"] = float(count)
    metrics.update(
        _compute_generation_byte_evidence_metrics(
            source_values=source_values,
            complete_values=complete_values,
            validated_values=validated_values,
            generation_attempted_values=generation_attempted_values,
            generation_complete_values=generation_complete_values,
            generation_validated_values=generation_validated_values,
        )
    )

    summary_text = (
        f"status={dict(status_counter)} "
        f"complete={dict(complete_counter)} "
        f"source={dict(source_counter)} "
        f"evidence_kind={dict(evidence_kind_counter)} "
        f"validated={dict(validated_counter)} "
        f"generation_byte_evidence="
        f"{{hit_rate={metrics['opd_generation_byte_evidence_hit_rate']:.4f}, "
        f"incomplete_rate={metrics['opd_generation_byte_evidence_incomplete_rate']:.4f}, "
        f"invalid_rate={metrics['opd_generation_byte_evidence_invalid_rate']:.4f}}} "
        f"top_errors={error_counter.most_common(3)}"
    )
    return metrics, summary_text


def summarize_opd_alignment_from_samples(samples: list[Any]) -> tuple[dict[str, float], str | None]:
    return summarize_opd_alignment(
        status_values=[getattr(sample, "opd_student_alignment_status", None) for sample in samples],
        complete_values=[getattr(sample, "opd_student_alignment_complete", None) for sample in samples],
        source_values=[getattr(sample, "opd_student_alignment_source", None) for sample in samples],
        validated_values=[getattr(sample, "opd_student_alignment_validated", None) for sample in samples],
        error_values=[getattr(sample, "opd_student_alignment_error", None) for sample in samples],
        metadata_values=[getattr(sample, "opd_student_alignment_metadata", None) for sample in samples],
        generation_attempted_values=[
            getattr(sample, "opd_generation_byte_evidence_attempted", None) for sample in samples
        ],
        generation_complete_values=[getattr(sample, "opd_generation_byte_evidence_complete", None) for sample in samples],
        generation_validated_values=[
            getattr(sample, "opd_generation_byte_evidence_validated", None) for sample in samples
        ],
    )


def summarize_opd_alignment_from_rollout_data(rollout_data: dict[str, Any]) -> tuple[dict[str, float], str | None]:
    return summarize_opd_alignment(
        status_values=rollout_data.get("opd_student_alignment_status_list"),
        complete_values=rollout_data.get("opd_student_alignment_complete_list"),
        source_values=rollout_data.get("opd_student_alignment_source_list"),
        validated_values=rollout_data.get("opd_student_alignment_validated_list"),
        error_values=rollout_data.get("opd_student_alignment_error_list"),
        metadata_values=rollout_data.get("opd_student_alignment_metadata_list"),
        generation_attempted_values=rollout_data.get("opd_generation_byte_evidence_attempted_list"),
        generation_complete_values=rollout_data.get("opd_generation_byte_evidence_complete_list"),
        generation_validated_values=rollout_data.get("opd_generation_byte_evidence_validated_list"),
    )
