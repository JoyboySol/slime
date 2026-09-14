"""Merge the complete AIME-like JSONL directory into one traceable JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--output-name", default="aime_like_all.jsonl")
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    total = 0
    seen_queries: set[str] = set()

    with output_path.open("w", encoding="utf-8") as out:
        for source_path in files:
            with source_path.open(encoding="utf-8") as src:
                for line_number, line in enumerate(src, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    query = record.get("query")
                    if query is None:
                        raise ValueError(f"Missing query in {source_path}:{line_number}")
                    if query in seen_queries:
                        raise ValueError(f"Duplicate query in {source_path}:{line_number}")
                    seen_queries.add(query)
                    record["merge_source_file"] = source_path.name
                    record["merge_source_line"] = line_number
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1

    manifest = {
        "input_dir": str(args.input_dir),
        "source_files": [p.name for p in files],
        "total_records": total,
        "unique_queries": len(seen_queries),
        "output": str(output_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
