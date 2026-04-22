from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

DEFAULT_STEP_KEYS = ("train/step", "rollout/step", "eval/step")


def _is_numeric_step_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_wandb_loggable_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (int, float, bool, str))


def filter_loggable_history_row(row: dict[str, Any]) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith("_"):
            continue
        if value is None:
            continue
        if not is_wandb_loggable_scalar(value):
            continue
        filtered[key] = value
    return filtered


def normalize_step_targets(step_targets: dict[str, Any] | None) -> dict[str, float]:
    if not step_targets:
        return {}

    normalized: dict[str, float] = {}
    for key, value in step_targets.items():
        if not _is_numeric_step_value(value):
            continue
        normalized[key] = float(value)
    return normalized


def build_uniform_step_targets(step_keys: Iterable[str], value: float | None) -> dict[str, float]:
    if value is None:
        return {}
    return {key: float(value) for key in step_keys}


def apply_step_offsets(row: dict[str, Any], step_offsets: dict[str, float]) -> dict[str, Any]:
    if not step_offsets:
        return dict(row)

    adjusted = dict(row)
    for key, offset in step_offsets.items():
        value = adjusted.get(key)
        if _is_numeric_step_value(value):
            adjusted[key] = value + offset
    return adjusted


def row_within_max_steps(row: dict[str, Any], max_step_targets: dict[str, float]) -> bool:
    for key, max_value in max_step_targets.items():
        value = row.get(key)
        if _is_numeric_step_value(value) and float(value) > max_value:
            return False
    return True


def trim_history_rows_to_max_steps(rows: Iterable[dict[str, Any]], max_step_targets: dict[str, float]) -> list[dict[str, Any]]:
    if not max_step_targets:
        return [dict(row) for row in rows]
    return [dict(row) for row in rows if row_within_max_steps(row, max_step_targets)]


def drop_first_history_rows(rows: Iterable[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    if count <= 0:
        return materialized
    return materialized[count:]


def repair_low_step_rows(
    rows: Iterable[dict[str, Any]],
    *,
    step_key: str,
    min_expected_step: float,
    anchor_step_keys: Iterable[str],
) -> list[dict[str, Any]]:
    repaired_rows: list[dict[str, Any]] = []
    latest_anchor_step: float | None = None
    normalized_anchor_keys = tuple(anchor_step_keys)

    for row in rows:
        repaired = dict(row)
        anchor_values = [
            float(repaired[key])
            for key in normalized_anchor_keys
            if key in repaired and _is_numeric_step_value(repaired[key])
        ]
        if anchor_values:
            latest_anchor_step = max(anchor_values)

        current_step = repaired.get(step_key)
        if (
            _is_numeric_step_value(current_step)
            and float(current_step) < float(min_expected_step)
            and latest_anchor_step is not None
            and latest_anchor_step >= float(min_expected_step)
        ):
            repaired[step_key] = latest_anchor_step
        repaired_rows.append(repaired)

    return repaired_rows


def shift_step_rows(
    rows: Iterable[dict[str, Any]],
    *,
    step_key: str,
    min_step_inclusive: float,
    delta: float,
) -> list[dict[str, Any]]:
    shifted_rows: list[dict[str, Any]] = []
    for row in rows:
        shifted = dict(row)
        current_step = shifted.get(step_key)
        if _is_numeric_step_value(current_step) and float(current_step) >= float(min_step_inclusive):
            shifted[step_key] = float(current_step) + float(delta)
        shifted_rows.append(shifted)
    return shifted_rows


def compute_first_step_values(rows: Iterable[dict[str, Any]], step_keys: Iterable[str]) -> dict[str, float]:
    first_values: dict[str, float] = {}
    pending = list(step_keys)
    if not pending:
        return first_values

    for row in rows:
        for key in pending[:]:
            value = row.get(key)
            if _is_numeric_step_value(value):
                first_values[key] = float(value)
                pending.remove(key)
        if not pending:
            break
    return first_values


def update_step_targets(
    current_targets: dict[str, float],
    rows: Iterable[dict[str, Any]],
    step_keys: Iterable[str],
) -> dict[str, float]:
    next_targets = dict(current_targets)
    max_values: dict[str, float] = {}
    for row in rows:
        for key in step_keys:
            value = row.get(key)
            if _is_numeric_step_value(value):
                max_values[key] = max(float(value), max_values.get(key, float("-inf")))

    for key, max_value in max_values.items():
        next_targets[key] = max_value + 1.0
    return next_targets


def compute_segment_step_offsets(
    rows: list[dict[str, Any]],
    *,
    next_targets: dict[str, float],
    step_keys: Iterable[str],
    manual_offsets: dict[str, float] | None = None,
) -> dict[str, float]:
    offsets = dict(manual_offsets or {})
    first_values = compute_first_step_values(rows, step_keys)
    for key in step_keys:
        if key in offsets:
            continue
        if key in next_targets and key in first_values:
            offsets[key] = next_targets[key] - first_values[key]
    return offsets


@dataclass(frozen=True)
class RunHistorySegment:
    run_path: str
    rows: list[dict[str, Any]]

    @property
    def run_id(self) -> str:
        return self.run_path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class MergedHistory:
    rows: list[dict[str, Any]]
    next_step_targets: dict[str, float]
    per_run_offsets: dict[str, dict[str, float]]


def merge_run_histories(
    segments: list[RunHistorySegment],
    *,
    step_keys: Iterable[str] = DEFAULT_STEP_KEYS,
    manual_offsets_by_run: dict[str, dict[str, float]] | None = None,
    max_step_targets: dict[str, float] | None = None,
) -> MergedHistory:
    normalized_step_keys = tuple(step_keys)
    normalized_max_step_targets = normalize_step_targets(max_step_targets)
    step_targets: dict[str, float] = {}
    merged_rows: list[dict[str, Any]] = []
    per_run_offsets: dict[str, dict[str, float]] = {}

    for segment in segments:
        bounded_rows = trim_history_rows_to_max_steps(segment.rows, normalized_max_step_targets)
        loggable_rows = [filter_loggable_history_row(row) for row in bounded_rows]
        loggable_rows = [row for row in loggable_rows if row]
        manual_offsets = (manual_offsets_by_run or {}).get(segment.run_id, {})
        offsets = compute_segment_step_offsets(
            loggable_rows,
            next_targets=step_targets,
            step_keys=normalized_step_keys,
            manual_offsets=manual_offsets,
        )
        per_run_offsets[segment.run_id] = offsets

        adjusted_rows = [apply_step_offsets(row, offsets) for row in loggable_rows]
        merged_rows.extend(adjusted_rows)
        step_targets = update_step_targets(step_targets, adjusted_rows, normalized_step_keys)

    return MergedHistory(
        rows=merged_rows,
        next_step_targets=step_targets,
        per_run_offsets=per_run_offsets,
    )


class RuntimeStepAligner:
    def __init__(self, step_targets: dict[str, Any] | None = None):
        self._step_targets = normalize_step_targets(step_targets)
        self._resolved_offsets: dict[str, float] = {}

    def apply(self, metrics: dict[str, Any]) -> dict[str, Any]:
        if not self._step_targets:
            return metrics

        adjusted = dict(metrics)
        for key, target in self._step_targets.items():
            value = adjusted.get(key)
            if not _is_numeric_step_value(value):
                continue
            if key not in self._resolved_offsets:
                self._resolved_offsets[key] = target - float(value)
            adjusted[key] = float(value) + self._resolved_offsets[key]
        return adjusted
