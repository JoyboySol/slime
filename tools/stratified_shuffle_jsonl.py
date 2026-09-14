"""Deterministically stratified-shuffle a JSONL dataset without changing rows."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("judge_answer_kind", "unknown")),
            str(record.get("merge_source_file", "unknown")),
        )
        buckets[key].append(record)

    rng = random.Random(args.seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    keys = sorted(buckets)
    emitted = {key: 0 for key in keys}
    output_records: list[dict] = []
    total = len(records)
    for position in range(total):
        # Pick the bucket most behind its proportional target at this point.
        # This preserves exact bucket counts while spreading every stratum.
        best = []
        best_deficit = None
        for key in keys:
            if not buckets[key]:
                continue
            target = (position + 1) * len(buckets[key]) / total
            deficit = target - emitted[key]
            if best_deficit is None or deficit > best_deficit + 1e-12:
                best = [key]
                best_deficit = deficit
            elif abs(deficit - best_deficit) <= 1e-12:
                best.append(key)
        key = rng.choice(best)
        output_records.append(buckets[key].pop())
        emitted[key] += 1

    if not args.in_place:
        output_path = args.input.with_name(args.input.stem + "_shuffled" + args.input.suffix)
    else:
        output_path = args.input.with_name(args.input.name + ".tmp")
        backup_path = args.input.with_name(args.input.name + ".before_shuffle")
        if not backup_path.exists():
            shutil.copy2(args.input, backup_path)

    with output_path.open("w", encoding="utf-8") as out:
        for record in output_records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.in_place:
        output_path.replace(args.input)
        output_path = args.input

    print(json.dumps({"output": str(output_path), "records": total, "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
