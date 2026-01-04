#!/usr/bin/env python3
"""
Build ONE JSONL file from EmpatheticDialogues (train/valid/test merged),
keeping ONLY:
  - conv_id (string)
  - utterance_idx (int)
  - prompt (string)
  - speaker_idx (int)
  - utterance (string)

Usage:
  python build_ed_jsonl.py --out data/empathetic_dialogues.jsonl
  python build_ed_jsonl.py --out data/ed.jsonl --cache_dir .cache/ed
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterator

URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"

SPLIT_TO_CSV = {
    "train": "empatheticdialogues/train.csv",
    "validation": "empatheticdialogues/valid.csv",
    "test": "empatheticdialogues/test.csv",
}


def download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    print(f"Downloading {url} -> {dst}", file=sys.stderr)
    urllib.request.urlretrieve(url, dst)  # nosec: trusted dataset host
    return dst


def iter_csv_from_tar(tar_path: Path, member_path: str) -> Iterator[Dict[str, str]]:
    """Stream CSV rows from a file inside the .tar.gz (no extraction)."""
    with tarfile.open(tar_path, mode="r:gz") as tar:
        member = tar.getmember(member_path)
        f = tar.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"Could not read {member_path} inside {tar_path}")

        text = io.TextIOWrapper(f, encoding="utf-8", newline="")
        reader = csv.DictReader(text)
        yield from reader


def normalize_row(row: Dict[str, str]) -> Dict[str, object]:
    def s(x: str | None) -> str:
        return (x or "").strip()

    def i(x: str | None) -> int:
        x = s(x)
        return int(x) if x else 0

    return {
        "conv_id": s(row.get("conv_id")),
        "utterance_idx": i(row.get("utterance_idx")),
        "prompt": s(row.get("prompt")),
        "speaker_idx": i(row.get("speaker_idx")),
        "utterance": s(row.get("utterance")),
    }


def iter_all_rows(tar_path: Path) -> Iterator[Dict[str, object]]:
    # Merge all splits into one stream (order: train -> validation -> test)
    for _split, member_path in SPLIT_TO_CSV.items():
        for row in iter_csv_from_tar(tar_path, member_path):
            yield normalize_row(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a merged EmpatheticDialogues JSONL (minimal fields).")
    p.add_argument("--out", required=True, type=Path, help="Output JSONL path (e.g. data/empathetic_dialogues.jsonl)")
    p.add_argument("--url", default=URL, help="Dataset URL (default: official FB public file)")
    p.add_argument("--cache_dir", type=Path, default=Path(".cache/empathetic_dialogues"), help="Cache dir for archive")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    archive_path = args.cache_dir / "empatheticdialogues.tar.gz"
    download(args.url, archive_path)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for obj in iter_all_rows(archive_path):
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
