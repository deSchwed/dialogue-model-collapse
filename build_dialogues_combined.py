#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

ED_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"

ED_SPLIT_TO_CSV = {
    "train": "empatheticdialogues/train.csv",
    "validation": "empatheticdialogues/valid.csv",
    "test": "empatheticdialogues/test.csv",
}


def normalize_text(s: str) -> str:
    return (s or "").replace("_comma_", ",").strip()


# ------------------------- DailyDialog -------------------------


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def convert_dailydialog_minimal(dailydialog_json: Path, prefix: str = "dd:") -> List[Dict[str, Any]]:
    data = load_json(dailydialog_json)
    if not isinstance(data, list):
        raise ValueError("DailyDialog input must be a JSON list of dialogues.")

    out: List[Dict[str, Any]] = []
    for d in data:
        conv_id = str(d.get("dialogue_id") or d.get("original_id") or "").strip()
        if not conv_id:
            continue

        turns_in = d.get("turns", [])
        if not isinstance(turns_in, list) or not turns_in:
            continue

        turns: List[Dict[str, Any]] = []
        for t in turns_in:
            speaker = (t.get("speaker") or "").strip().lower()
            if speaker not in ("user", "system"):
                speaker = "system"

            utt = normalize_text(str(t.get("utterance") or ""))
            if not utt:
                continue

            utt_idx = t.get("utt_idx")
            try:
                utt_idx = int(utt_idx)
            except Exception:
                utt_idx = len(turns)

            turns.append({"utt_idx": utt_idx, "speaker": speaker, "utterance": utt})

        if len(turns) < 2:
            continue

        turns.sort(key=lambda x: x["utt_idx"])
        out.append({"conv_id": f"{prefix}{conv_id}", "turns": turns})

    return out


# ---------------------- EmpatheticDialogues ----------------------


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


def parse_int(x: Any, default: int = 0) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default


def load_empathetic_dialogues_conversations(
    cache_dir: Path,
    url: str = ED_URL,
    prefix: str = "ed:",
) -> List[Dict[str, Any]]:
    """
    Builds minimal conversation objects:
      {"conv_id": "...", "turns":[{"utt_idx": int, "speaker": "user|system", "utterance": str}, ...]}
    """
    archive_path = cache_dir / "empatheticdialogues.tar.gz"
    download(url, archive_path)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # Stream all splits into one grouping by conv_id
    for _split, member_path in ED_SPLIT_TO_CSV.items():
        for row in iter_csv_from_tar(archive_path, member_path):
            conv_id = str(row.get("conv_id") or "").strip()
            if not conv_id:
                continue

            utt = normalize_text(str(row.get("utterance") or ""))
            if not utt:
                continue

            utt_idx = parse_int(row.get("utterance_idx"), default=10**9)
            spk_raw = row.get("speaker_idx")
            spk = parse_int(spk_raw, default=-1)

            grouped[conv_id].append(
                {
                    "utt_idx": utt_idx,
                    "speaker_idx": spk,
                    "utterance": utt,
                }
            )

    out: List[Dict[str, Any]] = []
    for conv_id, rows in grouped.items():
        rows.sort(key=lambda r: r["utt_idx"])

        # Map first distinct speaker_idx -> user, second -> system; anything else -> system
        speaker_map: Dict[int, str] = {}

        def map_role(spk: int) -> str:
            if spk not in speaker_map:
                if len(speaker_map) == 0:
                    speaker_map[spk] = "user"
                elif len(speaker_map) == 1:
                    speaker_map[spk] = "system"
                else:
                    speaker_map[spk] = "system"
            return speaker_map[spk]

        turns: List[Dict[str, Any]] = []
        for r in rows:
            turns.append(
                {
                    "utt_idx": int(r["utt_idx"]),
                    "speaker": map_role(int(r["speaker_idx"])),
                    "utterance": r["utterance"],
                }
            )

        if len(turns) < 2:
            continue

        out.append({"conv_id": f"{prefix}{conv_id}", "turns": turns})

    return out


# ------------------------- Renumbering -------------------------


def renumber_conversations(convs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic renumbering:
      - sort by existing conv_id
      - conv_id -> conv_000000, conv_000001, ...
      - utt_idx -> 0..T-1 after sorting
    """
    convs_sorted = sorted(convs, key=lambda c: str(c.get("conv_id", "")))

    renumbered: List[Dict[str, Any]] = []
    for i, c in enumerate(convs_sorted):
        turns = list(c.get("turns") or [])
        turns.sort(key=lambda t: int(t.get("utt_idx", 10**9)))

        new_turns = []
        for j, t in enumerate(turns):
            utt = normalize_text(str(t.get("utterance") or ""))
            if not utt:
                continue
            spk = (t.get("speaker") or "system").strip().lower()
            if spk not in ("user", "system"):
                spk = "system"
            new_turns.append({"utt_idx": j, "speaker": spk, "utterance": utt})

        if len(new_turns) < 2:
            continue

        renumbered.append({"conv_id": f"conv_{i:06d}", "turns": new_turns})

    return renumbered


# ------------------------- Main -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a single minimal conversation JSON by combining DailyDialog + EmpatheticDialogues."
    )
    p.add_argument(
        "--dailydialog_json",
        type=Path,
        required=True,
        help="Path to DailyDialog JSON list (your merged ConvLab DailyDialog file).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/raw"),
        help='Output directory. The script will write "dialogues_combined.json" there.',
    )
    p.add_argument(
        "--ed_cache_dir",
        type=Path,
        default=Path(".cache/empathetic_dialogues"),
        help="Cache dir for EmpatheticDialogues archive download.",
    )
    p.add_argument("--ed_url", type=str, default=ED_URL, help="EmpatheticDialogues tar.gz URL.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dailydialog_json.exists():
        raise FileNotFoundError(f"DailyDialog JSON not found: {args.dailydialog_json}")

    out_json = args.out_dir / "dialogues_combined.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dd = convert_dailydialog_minimal(args.dailydialog_json, prefix="dd:")
    ed = load_empathetic_dialogues_conversations(args.ed_cache_dir, url=args.ed_url, prefix="ed:")

    combined = dd + ed
    combined = renumber_conversations(combined)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"DailyDialog conversations: {len(dd)}")
    print(f"EmpatheticDialogues conversations: {len(ed)}")
    print(f"Combined conversations (after renumber/filter): {len(combined)}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
