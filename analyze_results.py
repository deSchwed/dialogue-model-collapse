#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

LOSS_DICT_RE = re.compile(r"^\s*\{.*('loss'|\"loss\")\s*:\s*.*\}\s*$")
LABEL_FROM_METRIC_RE = re.compile(r"^gen(?P<gen>\d+)_r(?P<r>\d{3})$")


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def label_to_gen_ratio(label: str) -> Tuple[int, float, str]:
    """
    label examples:
      - gen00
      - gen01_r025, gen02_r050, gen03_r100
    """
    if label == "gen00":
        return 0, 0.0, "r000"

    m = LABEL_FROM_METRIC_RE.match(label)
    if m:
        g = int(m.group("gen"))
        r = int(m.group("r")) / 100.0
        return g, r, f"r{int(r * 100):03d}"

    # fallback
    mg = re.search(r"gen(\d+)", label)
    mr = re.search(r"r(\d{3})", label)
    if mg and mr:
        g = int(mg.group(1))
        r = int(mr.group(1)) / 100.0
        return g, r, f"r{int(r * 100):03d}"

    return -1, float("nan"), "r???"


def _ema_smooth(y: List[float], alpha: float = 0.25) -> List[float]:
    """Exponential moving average smoothing."""
    if not y:
        return y
    out = [y[0]]
    for v in y[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out

# -----------------------------
# Collect eval metrics
# -----------------------------

def collect_eval_metrics(metrics_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    all_keys = set(["label", "generation", "ratio", "ratio_label"])

    for p in sorted(metrics_dir.glob("*.json")):
        label = p.stem
        gen, ratio, ratio_label = label_to_gen_ratio(label)
        try:
            d = load_json(p)
        except Exception:
            continue

        row = {"label": label, "generation": gen, "ratio": ratio, "ratio_label": ratio_label}
        if isinstance(d, dict):
            for k, v in d.items():
                row[k] = v
                all_keys.add(k)
        rows.append(row)

    # Add deltas vs baseline if baseline exists
    baseline = next((r for r in rows if r["label"] == "gen00"), None)
    if baseline:
        base_ppl = safe_float(baseline.get("perplexity_real_test"))
        base_d1  = safe_float(baseline.get("distinct_1"))
        base_d2  = safe_float(baseline.get("distinct_2"))
        base_rep = safe_float(baseline.get("repetition_4gram"))

        for r in rows:
            ppl = safe_float(r.get("perplexity_real_test"))
            d1  = safe_float(r.get("distinct_1"))
            d2  = safe_float(r.get("distinct_2"))
            rep = safe_float(r.get("repetition_4gram"))

            if base_ppl is not None and ppl is not None:
                r["ppl_delta"] = ppl - base_ppl
                r["ppl_ratio"] = ppl / base_ppl if base_ppl != 0 else ""
                all_keys.update(["ppl_delta", "ppl_ratio"])
            if base_d1 is not None and d1 is not None:
                r["distinct_1_delta"] = d1 - base_d1
                all_keys.add("distinct_1_delta")
            if base_d2 is not None and d2 is not None:
                r["distinct_2_delta"] = d2 - base_d2
                all_keys.add("distinct_2_delta")
            if base_rep is not None and rep is not None:
                r["repetition_4gram_delta"] = rep - base_rep
                all_keys.add("repetition_4gram_delta")

    preferred = [
        "label", "generation", "ratio", "ratio_label",
        "perplexity_real_test", "distinct_1", "distinct_2", "repetition_4gram",
        "ppl_delta", "ppl_ratio", "distinct_1_delta", "distinct_2_delta", "repetition_4gram_delta",
    ]
    others = sorted([k for k in all_keys if k not in preferred])
    fieldnames = [k for k in preferred if k in all_keys] + others

    rows = sorted(rows, key=lambda r: (int(r.get("generation", -1)), float(r.get("ratio", math.nan))))
    return rows, fieldnames


# -----------------------------
# Training loss from logs/*_train_loss.txt
# -----------------------------

@dataclass
class LossPoint:
    label: str
    epoch: float
    loss: float
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None


def parse_loss_dict_line(line: str) -> Optional[Dict[str, Any]]:
    s = line.strip()
    if not s or not LOSS_DICT_RE.match(s):
        return None
    try:
        # your files use python dict repr with single quotes
        if s.startswith("{'") or "':" in s:
            return ast.literal_eval(s)
        # fallback
        return json.loads(s)
    except Exception:
        return None


def collect_train_loss_files(logs_dir: Path) -> Tuple[List[LossPoint], List[Dict[str, Any]]]:
    points: List[LossPoint] = []
    rows: List[Dict[str, Any]] = []

    for p in sorted(logs_dir.glob("*_train_loss.txt")):
        label = p.stem.replace("_train_loss", "")  # gen00, gen01_r025, ...
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                d = parse_loss_dict_line(line)
                if not d:
                    continue
                if "loss" not in d or "epoch" not in d:
                    continue
                lp = LossPoint(
                    label=label,
                    epoch=float(d["epoch"]),
                    loss=float(d["loss"]),
                    learning_rate=safe_float(d.get("learning_rate")),
                    grad_norm=safe_float(d.get("grad_norm")),
                )
                points.append(lp)
                rows.append({
                    "label": label,
                    "epoch": lp.epoch,
                    "loss": lp.loss,
                    "learning_rate": lp.learning_rate if lp.learning_rate is not None else "",
                    "grad_norm": lp.grad_norm if lp.grad_norm is not None else "",
                    "source_file": str(p),
                })

    return points, rows


# -----------------------------
# Plotting
# -----------------------------

def plot_metric_by_ratio(
    eval_rows: List[Dict[str, Any]],
    metric_key: str,
    out_png: Path,
    title: str,
    ylabel: str,
) -> None:
    by_ratio: Dict[str, List[Dict[str, Any]]] = {}
    for r in eval_rows:
        if metric_key not in r:
            continue
        gen = r.get("generation", -1)
        if gen is None or gen == -1:
            continue
        by_ratio.setdefault(r.get("ratio_label", "r???"), []).append(r)

    plt.figure()
    for ratio_label, rows in sorted(by_ratio.items()):
        rows = sorted(rows, key=lambda x: int(x["generation"]))
        xs = [int(x["generation"]) for x in rows]
        ys = []
        for x in rows:
            v = x.get(metric_key)
            try:
                ys.append(float(v))
            except Exception:
                ys.append(float("nan"))
        plt.plot(xs, ys, marker="o", label=ratio_label)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_loss_curves(
    loss_points: List[LossPoint],
    out_png: Path,
    title: str,
    max_labels: int = 12,
) -> None:
    """
    Plots up to max_labels curves
    Preference: gen00 + latest generation per ratio if available.
    """
    if not loss_points:
        return

    # group
    by_label: Dict[str, List[LossPoint]] = {}
    for lp in loss_points:
        by_label.setdefault(lp.label, []).append(lp)

    # choose labels
    labels: List[str] = []
    if "gen00" in by_label:
        labels.append("gen00")

    candidates: Dict[str, Tuple[int, str]] = {}  # ratio_label -> (gen, label)
    for lab in by_label.keys():
        if lab == "gen00":
            continue
        g, r, rl = label_to_gen_ratio(lab)
        if g < 0:
            continue
        prev = candidates.get(rl)
        if prev is None or g > prev[0]:
            candidates[rl] = (g, lab)

    for rl, (_g, lab) in sorted(candidates.items()):
        labels.append(lab)

    # fill remaining up to max_labels
    if len(labels) < max_labels:
        for lab in sorted(by_label.keys()):
            if lab not in labels:
                labels.append(lab)
            if len(labels) >= max_labels:
                break

    # ---- Plot ----
    plt.figure(figsize=(16, 9))  # bigger plot

    for lab in labels:
        pts = sorted(by_label[lab], key=lambda p: p.epoch)
        xs = [p.epoch for p in pts]
        ys = [p.loss for p in pts]

        # Smoothing
        ys_smooth = _ema_smooth(ys, alpha=0.25)

        plt.plot(xs, ys_smooth, linewidth=2.2, label=lab)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.grid(True, alpha=0.25)

    # Put legend outside to the right
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    plt.tight_layout(rect=[0, 0, 0.80, 1])  # leave space for legend

    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=250)
    plt.close()

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("paper_assets"))
    ap.add_argument("--metrics_dir", type=Path, default=Path("metrics"))
    ap.add_argument("--logs_dir", type=Path, default=Path("logs"))
    args = ap.parse_args()

    metrics_dir = args.metrics_dir
    logs_dir = args.logs_dir
    out_dir = args.out_dir
    plots_dir = out_dir / "plots"

    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    # Eval metrics
    eval_rows, eval_fields = collect_eval_metrics(metrics_dir) if metrics_dir.exists() else ([], [])
    if eval_rows:
        write_csv(out_dir / "eval_metrics.csv", eval_rows, eval_fields)
        dump_json(out_dir / "eval_metrics.json", eval_rows)

        plot_metric_by_ratio(
            eval_rows,
            metric_key="perplexity_real_test",
            out_png=plots_dir / "perplexity_vs_generation.png",
            title="Perplexity on real test vs generation",
            ylabel="Perplexity (lower is better)",
        )
        plot_metric_by_ratio(
            eval_rows,
            metric_key="distinct_1",
            out_png=plots_dir / "distinct1_vs_generation.png",
            title="Distinct-1 vs generation",
            ylabel="Distinct-1 (higher is more diverse)",
        )
        plot_metric_by_ratio(
            eval_rows,
            metric_key="distinct_2",
            out_png=plots_dir / "distinct2_vs_generation.png",
            title="Distinct-2 vs generation",
            ylabel="Distinct-2 (higher is more diverse)",
        )
        plot_metric_by_ratio(
            eval_rows,
            metric_key="repetition_4gram",
            out_png=plots_dir / "repetition4gram_vs_generation.png",
            title="4-gram repetition vs generation",
            ylabel="4-gram repetition (lower is better)",
        )

    # Training loss from *_train_loss.txt files
    loss_points, loss_rows = collect_train_loss_files(logs_dir) if logs_dir.exists() else ([], [])
    if loss_rows:
        write_csv(
            out_dir / "train_loss_history.csv",
            loss_rows,
            ["label", "epoch", "loss", "learning_rate", "grad_norm", "source_file"],
        )
        plot_loss_curves(
            loss_points,
            out_png=plots_dir / "loss_curves.png",
            title="Training loss curves",
            max_labels=14,
        )

    summary = {
        "found_eval_files": len(eval_rows),
        "found_loss_points": len(loss_points),
        "outputs": {
            "eval_metrics_csv": str((out_dir / "eval_metrics.csv").resolve()) if eval_rows else None,
            "train_loss_csv": str((out_dir / "train_loss_history.csv").resolve()) if loss_rows else None,
            "plots_dir": str(plots_dir.resolve()),
        },
        "notes": [
            "Eval plots use metrics/*.json.",
            "Loss curves use logs/*_train_loss.txt (python dict lines with loss/epoch).",
        ],
    }
    dump_json(out_dir / "summary.json", summary)

    print("Done. Wrote:", out_dir)


if __name__ == "__main__":
    main()
