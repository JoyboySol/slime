#!/usr/bin/env python3

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


def load_rollout(path: str) -> dict[str, Any]:
    rollout = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rollout, dict):
        raise TypeError(f"Expected rollout file to contain a dict, got {type(rollout).__name__}")
    if "samples" not in rollout or not isinstance(rollout["samples"], list):
        raise KeyError("Rollout file must contain a list under the 'samples' key")
    return rollout


def get_numeric_reward(sample: dict[str, Any]) -> float | None:
    reward = sample.get("reward")
    if isinstance(reward, (int, float)):
        return float(reward)
    return None


def summarize_statuses(samples: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for sample in samples:
        status = str(sample.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def format_stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} "
        f"mean={statistics.mean(values):.3f} "
        f"median={statistics.median(values):.3f} "
        f"min={min(values):.3f} "
        f"max={max(values):.3f}"
    )


def print_summary(rollout: dict[str, Any]) -> None:
    samples = rollout["samples"]
    response_lengths = [float(sample.get("response_length", 0)) for sample in samples]
    numeric_rewards = [reward for sample in samples if (reward := get_numeric_reward(sample)) is not None]

    print(f"rollout_id: {rollout.get('rollout_id')}")
    print(f"num_samples: {len(samples)}")
    print(f"statuses: {summarize_statuses(samples)}")
    print(f"response_length: {format_stats(response_lengths)}")
    print(f"numeric_reward: {format_stats(numeric_rewards)}")


def filter_samples(samples: list[dict[str, Any]], only_truncated: bool) -> list[dict[str, Any]]:
    if not only_truncated:
        return list(samples)
    return [sample for sample in samples if sample.get("status") == "truncated"]


def sort_key(sample: dict[str, Any], field: str) -> Any:
    if field == "reward":
        reward = get_numeric_reward(sample)
        return float("-inf") if reward is None else reward
    return sample.get(field)


def print_table(
    rollout: dict[str, Any],
    *,
    limit: int | None,
    only_truncated: bool,
    sort_by: str,
    descending: bool,
) -> None:
    samples = filter_samples(rollout["samples"], only_truncated=only_truncated)
    samples.sort(key=lambda sample: sort_key(sample, sort_by), reverse=descending)
    if limit is not None:
        samples = samples[:limit]

    print("index\tgroup_index\tstatus\tresponse_length\treward")
    for sample in samples:
        reward = sample.get("reward")
        print(
            "\t".join(
                [
                    str(sample.get("index")),
                    str(sample.get("group_index")),
                    str(sample.get("status")),
                    str(sample.get("response_length")),
                    json.dumps(reward, ensure_ascii=True, sort_keys=True) if isinstance(reward, dict) else str(reward),
                ]
            )
        )


def get_sample_by_index(samples: list[dict[str, Any]], index: int | None, nth: int | None) -> dict[str, Any]:
    if index is not None:
        for sample in samples:
            if sample.get("index") == index:
                return sample
        raise ValueError(f"No sample with index={index}")
    if nth is None:
        raise ValueError("Either --index or --nth is required")
    if nth < 0 or nth >= len(samples):
        raise ValueError(f"--nth must be in [0, {len(samples) - 1}]")
    return samples[nth]


def print_show(rollout: dict[str, Any], *, index: int | None, nth: int | None) -> None:
    sample = get_sample_by_index(rollout["samples"], index=index, nth=nth)

    for key in [
        "group_index",
        "index",
        "status",
        "response_length",
        "reward",
        "label",
        "metadata",
        "teacher_log_probs",
        "rollout_log_probs",
    ]:
        print(f"{key}: {sample.get(key)}")
    print("--- prompt ---")
    print(sample.get("prompt", ""))
    print("--- response ---")
    print(sample.get("response", ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect saved slime rollout debug files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary", help="Print rollout-level summary statistics.")
    summary_parser.add_argument("--path", required=True, help="Path to rollout_*.pt or rollout_eval_*.pt")

    table_parser = subparsers.add_parser("table", help="Print a table of sample-level summaries.")
    table_parser.add_argument("--path", required=True, help="Path to rollout_*.pt or rollout_eval_*.pt")
    table_parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to print")
    table_parser.add_argument("--only-truncated", action="store_true", help="Only show truncated samples")
    table_parser.add_argument(
        "--sort-by",
        choices=["index", "group_index", "status", "response_length", "reward"],
        default="index",
        help="Field used to sort rows",
    )
    table_parser.add_argument("--descending", action="store_true", help="Sort in descending order")

    show_parser = subparsers.add_parser("show", help="Print the full details for one sample.")
    show_parser.add_argument("--path", required=True, help="Path to rollout_*.pt or rollout_eval_*.pt")
    show_group = show_parser.add_mutually_exclusive_group(required=True)
    show_group.add_argument("--index", type=int, help="Select by sample['index']")
    show_group.add_argument("--nth", type=int, help="Select the nth sample in the saved list")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rollout_path = Path(args.path)
    if not rollout_path.exists():
        parser.error(f"Rollout file does not exist: {rollout_path}")

    try:
        rollout = load_rollout(str(rollout_path))
        if args.command == "summary":
            print_summary(rollout)
        elif args.command == "table":
            print_table(
                rollout,
                limit=args.limit,
                only_truncated=args.only_truncated,
                sort_by=args.sort_by,
                descending=args.descending,
            )
        elif args.command == "show":
            print_show(rollout, index=args.index, nth=args.nth)
        else:
            parser.error(f"Unsupported command: {args.command}")
    except Exception as exc:  # pragma: no cover - surfaced through CLI tests
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
