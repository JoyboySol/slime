#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import wandb

from slime.utils.wandb_canonical import DEFAULT_STEP_KEYS, RunHistorySegment, merge_run_histories
from slime.utils.wandb_canonical import build_uniform_step_targets
from slime.utils.wandb_canonical import drop_first_history_rows, repair_low_step_rows
from slime.utils.wandb_canonical import shift_step_rows
from slime.utils.wandb_utils import _init_wandb_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple W&B runs into one canonical run.")
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument("--entity", default=None, help="W&B entity. Defaults to the logged-in default entity.")
    parser.add_argument(
        "--source-run",
        action="append",
        required=True,
        dest="source_runs",
        help="Source run id or full entity/project/run_id path. Pass multiple times in chronological order.",
    )
    parser.add_argument("--canonical-run-id", default=None, help="Canonical run id. Defaults to a generated id.")
    parser.add_argument("--canonical-group", default=None, help="Canonical run group.")
    parser.add_argument("--canonical-name", default=None, help="Canonical run name.")
    parser.add_argument("--wandb-dir", default=None, help="If set, persist slime_wandb_state.json here for future resumes.")
    parser.add_argument(
        "--history-source",
        choices=("auto", "local", "api"),
        default="auto",
        help="How to load source histories. Defaults to local .wandb files when available, otherwise W&B Public API.",
    )
    parser.add_argument(
        "--manual-step-offsets-json",
        default=None,
        help="Optional JSON file mapping run_id -> metric key -> offset delta.",
    )
    parser.add_argument(
        "--step-key",
        action="append",
        dest="step_keys",
        help="Metric keys treated as logical steps. Defaults to train/step, rollout/step, eval/step.",
    )
    parser.add_argument(
        "--max-safe-step",
        type=float,
        default=None,
        help="If set, ignore source history rows whose step metrics exceed this checkpoint-safe boundary.",
    )
    parser.add_argument(
        "--skip-first-rows-json",
        default=None,
        help="Optional JSON string or file mapping run_id -> number of leading history rows to skip.",
    )
    parser.add_argument(
        "--repair-low-step-json",
        default=None,
        help=(
            "Optional JSON string or file mapping run_id -> list of repair specs. "
            "Each spec needs step_key, min_expected_step, and anchor_step_keys."
        ),
    )
    parser.add_argument(
        "--shift-step-json",
        default=None,
        help=(
            "Optional JSON string or file mapping run_id -> list of shift specs. "
            "Each spec needs step_key, min_step_inclusive, and delta."
        ),
    )
    parser.add_argument(
        "--wandb-timeout",
        type=int,
        default=int(os.environ.get("WANDB_PUBLIC_API_TIMEOUT", "60")),
        help="Timeout in seconds for W&B Public API history scans.",
    )
    parser.add_argument("--wandb-key", default=os.environ.get("WANDB_API_KEY"), help="Optional W&B API key.")
    parser.add_argument("--wandb-host", default=os.environ.get("WANDB_HOST"), help="Optional custom W&B host.")
    parser.add_argument("--dry-run", action="store_true", help="Print the merge plan without creating the canonical run.")
    return parser.parse_args()


def _resolve_entity(args: argparse.Namespace, api: wandb.Api) -> str:
    entity = args.entity or api.default_entity
    if entity is None:
        raise SystemExit("Could not determine W&B entity. Pass --entity explicitly.")
    return entity


def _resolve_run_path(raw: str, *, entity: str, project: str) -> str:
    return raw if raw.count("/") == 2 else f"{entity}/{project}/{raw}"


def _load_manual_offsets(path: str | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    normalized: dict[str, dict[str, float]] = {}
    for run_id, offsets in payload.items():
        normalized[run_id] = {key: float(value) for key, value in offsets.items()}
    return normalized


def _load_json_arg(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if raw.lstrip().startswith("{"):
        return json.loads(raw)
    with open(raw, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_checkpoint_safe_step(args: argparse.Namespace) -> float | None:
    if args.max_safe_step is not None:
        return float(args.max_safe_step)

    if not args.wandb_dir:
        return None

    checkpoint_hint = Path(args.wandb_dir).resolve().parent / "actor_ckpt" / "latest_checkpointed_iteration.txt"
    if not checkpoint_hint.exists():
        return None

    raw = checkpoint_hint.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return float(raw)


def _apply_segment_row_transforms(
    segments: list[RunHistorySegment],
    *,
    skip_first_rows: dict[str, int],
    repair_low_step_specs: dict[str, list[dict[str, Any]]],
    shift_step_specs: dict[str, list[dict[str, Any]]],
) -> list[RunHistorySegment]:
    transformed: list[RunHistorySegment] = []
    for segment in segments:
        rows = list(segment.rows)
        rows = drop_first_history_rows(rows, int(skip_first_rows.get(segment.run_id, 0)))
        for spec in repair_low_step_specs.get(segment.run_id, []):
            rows = repair_low_step_rows(
                rows,
                step_key=spec["step_key"],
                min_expected_step=float(spec["min_expected_step"]),
                anchor_step_keys=spec["anchor_step_keys"],
            )
        for spec in shift_step_specs.get(segment.run_id, []):
            rows = shift_step_rows(
                rows,
                step_key=spec["step_key"],
                min_step_inclusive=float(spec["min_step_inclusive"]),
                delta=float(spec["delta"]),
            )
        transformed.append(RunHistorySegment(run_path=segment.run_path, rows=rows))
    return transformed


def _load_segments_from_api(api: wandb.Api, run_paths: list[str]) -> list[RunHistorySegment]:
    segments: list[RunHistorySegment] = []
    for run_path in run_paths:
        run = api.run(run_path)
        rows = list(run.scan_history())
        segments.append(RunHistorySegment(run_path=run_path, rows=rows))
    return segments


def _coerce_history_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _history_item_key(item: Any) -> str:
    if getattr(item, "key", ""):
        return str(item.key)
    nested_key = getattr(item, "nested_key", ())
    return ".".join(str(part) for part in nested_key)


def _row_signature(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=True, sort_keys=True)


def _extract_history_rows_from_local_wandb_file(wandb_file: Path) -> list[dict[str, Any]]:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    datastore = DataStore()
    datastore.open_for_scan(str(wandb_file))
    rows: list[dict[str, Any]] = []

    while True:
        data = datastore.scan_data()
        if data is None:
            break

        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue

        row: dict[str, Any] = {}
        for item in record.history.item:
            row[_history_item_key(item)] = _coerce_history_value(item.value_json)
        if row:
            rows.append(row)

    return rows


def _collect_local_wandb_files(wandb_dir: str, run_id: str) -> list[Path]:
    local_root = Path(wandb_dir) / "wandb"
    if not local_root.exists():
        return []
    return sorted(path for path in local_root.glob(f"run-*-{run_id}/run-{run_id}.wandb") if path.is_file())


def _load_segments_from_local_wandb(run_paths: list[str], *, wandb_dir: str) -> list[RunHistorySegment]:
    segments: list[RunHistorySegment] = []
    for run_path in run_paths:
        run_id = run_path.rsplit("/", 1)[-1]
        local_files = _collect_local_wandb_files(wandb_dir, run_id)
        if not local_files:
            raise FileNotFoundError(f"No local .wandb files found for run_id={run_id} under {wandb_dir}")

        rows: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        for wandb_file in local_files:
            for row in _extract_history_rows_from_local_wandb_file(wandb_file):
                signature = _row_signature(row)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                rows.append(row)
        segments.append(RunHistorySegment(run_path=run_path, rows=rows))
    return segments


def _load_segments(
    api: wandb.Api,
    run_paths: list[str],
    *,
    history_source: str,
    wandb_dir: str | None,
) -> tuple[list[RunHistorySegment], str]:
    if history_source != "api" and wandb_dir:
        all_local_files = {
            run_path: _collect_local_wandb_files(wandb_dir, run_path.rsplit("/", 1)[-1]) for run_path in run_paths
        }
        if history_source == "local":
            missing = [run_path for run_path, files in all_local_files.items() if not files]
            if missing:
                raise FileNotFoundError(f"Missing local .wandb files for runs: {missing}")
            return _load_segments_from_local_wandb(run_paths, wandb_dir=wandb_dir), "local"

        if all(all_local_files.values()):
            return _load_segments_from_local_wandb(run_paths, wandb_dir=wandb_dir), "local"

    return _load_segments_from_api(api, run_paths), "api"


def _build_state_payload(
    *,
    canonical_run_id: str,
    project: str,
    entity: str | None,
    group: str | None,
    name: str | None,
    next_step_targets: dict[str, float],
    source_runs: list[str],
) -> dict[str, Any]:
    return {
        "entity": entity,
        "group": group,
        "project": project,
        "run_id": canonical_run_id,
        "run_name": name,
        "source_runs": source_runs,
        "step_targets": next_step_targets,
    }


def _compute_resume_step_targets(
    *,
    merged_next_step_targets: dict[str, float],
    step_keys: tuple[str, ...],
    max_safe_step: float | None,
) -> dict[str, float]:
    if max_safe_step is None:
        return merged_next_step_targets
    return build_uniform_step_targets(step_keys, max_safe_step + 1.0)


def _persist_state(wandb_dir: str, payload: dict[str, Any]) -> Path:
    state_path = Path(wandb_dir) / "slime_wandb_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state_path


def main() -> None:
    args = parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    api = wandb.Api(timeout=args.wandb_timeout)
    entity = _resolve_entity(args, api)
    source_run_paths = [_resolve_run_path(raw, entity=entity, project=args.project) for raw in args.source_runs]
    segments, history_source_used = _load_segments(
        api,
        source_run_paths,
        history_source=args.history_source,
        wandb_dir=args.wandb_dir,
    )
    segments = _apply_segment_row_transforms(
        segments,
        skip_first_rows={k: int(v) for k, v in _load_json_arg(args.skip_first_rows_json).items()},
        repair_low_step_specs=_load_json_arg(args.repair_low_step_json),
        shift_step_specs=_load_json_arg(args.shift_step_json),
    )
    manual_offsets = _load_manual_offsets(args.manual_step_offsets_json)
    step_keys = tuple(args.step_keys or DEFAULT_STEP_KEYS)
    resolved_max_safe_step = _resolve_checkpoint_safe_step(args)
    max_step_targets = build_uniform_step_targets(step_keys, resolved_max_safe_step)
    merged = merge_run_histories(
        segments,
        step_keys=step_keys,
        manual_offsets_by_run=manual_offsets,
        max_step_targets=max_step_targets,
    )
    resume_step_targets = _compute_resume_step_targets(
        merged_next_step_targets=merged.next_step_targets,
        step_keys=step_keys,
        max_safe_step=resolved_max_safe_step,
    )

    canonical_run_id = args.canonical_run_id or wandb.util.generate_id()
    canonical_group = args.canonical_group or "canonical-" + (segments[-1].run_id if segments else canonical_run_id)
    canonical_name = args.canonical_name or canonical_group

    state_payload = _build_state_payload(
        canonical_run_id=canonical_run_id,
        project=args.project,
        entity=entity,
        group=canonical_group,
        name=canonical_name,
        next_step_targets=resume_step_targets,
        source_runs=source_run_paths,
    )

    print(json.dumps(
        {
            "canonical_run_id": canonical_run_id,
            "canonical_group": canonical_group,
            "canonical_name": canonical_name,
            "history_source_used": history_source_used,
            "source_runs": source_run_paths,
            "per_run_offsets": merged.per_run_offsets,
            "next_step_targets": merged.next_step_targets,
            "merged_row_count": len(merged.rows),
            "resume_step_targets": resume_step_targets,
            "resolved_max_safe_step": resolved_max_safe_step,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))

    if args.dry_run:
        return

    run = wandb.init(
        entity=entity,
        project=args.project,
        group=canonical_group,
        name=canonical_name,
        id=canonical_run_id,
        resume="never",
        dir=args.wandb_dir,
        config={
            "canonical_source_runs": source_run_paths,
            "canonical_step_keys": list(step_keys),
            "canonical_per_run_offsets": merged.per_run_offsets,
            "canonical_max_safe_step": resolved_max_safe_step,
        },
    )
    _init_wandb_common()
    for row in merged.rows:
        wandb.log(row)
    run.summary["canonical_source_runs"] = source_run_paths
    run.summary["canonical_next_step_targets"] = resume_step_targets
    run.finish()

    if args.wandb_dir:
        state_path = _persist_state(args.wandb_dir, state_payload)
        print(f"Persisted canonical W&B state to {state_path}")


if __name__ == "__main__":
    main()
