from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def iter_json_items(path: Path):
    """
    Supports:
      1) Top-level JSON list: [ {...}, {...}, ... ]
      2) JSONL: one JSON object per line
    """
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON is not a list.")
            for obj in data:
                yield obj
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def analyze(path: Path, sample_keys: int = 5) -> None:
    split_counts = Counter()
    dataset_counts = Counter()
    keys_counter = Counter()
    n = 0
    bad = 0

    for obj in iter_json_items(path):
        n += 1
        if not isinstance(obj, dict):
            bad += 1
            continue

        # Key stats
        for k in obj.keys():
            keys_counter[k] += 1

        # Most common split fields
        split = None
        for split_key in ("data_split", "split"):
            if split_key in obj:
                split = obj.get(split_key)
                break
        if split is not None:
            split_counts[str(split)] += 1

        # Optional dataset field
        if "dataset" in obj:
            dataset_counts[str(obj.get("dataset"))] += 1

    print(f"File: {path}")
    print(f"Total items read: {n}")
    if bad:
        print(f"Non-dict items: {bad}")

    print("\nTop-level keys (most frequent):")
    for k, c in keys_counter.most_common(sample_keys):
        print(f"  {k}: {c}/{n}")

    if split_counts:
        print("\nSplit distribution (from 'data_split' or 'split'):")
        for k, c in split_counts.most_common():
            print(f"  {k}: {c}")
    else:
        print("\nNo 'data_split' or 'split' field found at top level.")

    if dataset_counts:
        print("\nDataset field distribution:")
        for k, c in dataset_counts.most_common():
            print(f"  {k}: {c}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a dialogue JSON/JSONL file for split coverage.")
    ap.add_argument("path", type=Path, help="Path to .json (list) or .jsonl file.")
    ap.add_argument("--sample_keys", type=int, default=10, help="How many top-level keys to show.")
    args = ap.parse_args()
    analyze(args.path, sample_keys=args.sample_keys)


if __name__ == "__main__":
    main()
