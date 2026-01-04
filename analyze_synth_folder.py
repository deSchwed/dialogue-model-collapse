from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ----------------------------
# Config (no CLI args)
# ----------------------------
SYNTH_DIR_CANDIDATES = [
    Path("data") / "synth",  # your pipeline default
    Path("synth"),           # common alternative
    Path("data_synth"),      # fallback
]
OUT_DIR = Path("analysis_reports")
REPORT_STEM = "synth_large_report"  # -> .json, .txt
EXAMPLES_TXT = OUT_DIR / "examples.txt"

RNG_SEED = 12345
EXAMPLES_PER_FILE = 3
MAX_SCAN_EXAMPLES_PER_FILE = None  # set to an int for quick tests (e.g., 2000)


# ----------------------------
# Tokenization / helpers
# ----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]", re.UNICODE)

ROLE_ARTIFACT_RE = re.compile(
    r"(^|\n)\s*(system|user|assistant)\s*(:|\n)",
    re.IGNORECASE,
)
SYSTEM_PROMPT_SNIPPET_RE = re.compile(
    r"you are a helpful conversational agent",
    re.IGNORECASE,
)


def simple_tokens(text: str) -> List[str]:
    """Lightweight tokenizer: words + punctuation as tokens."""
    return _WORD_RE.findall(text)


def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile (0..100)."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = math.ceil((p / 100) * len(xs)) - 1
    k = max(0, min(k, len(xs) - 1))
    return xs[k]


def ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def repeated_ngram_rate(tokens: List[str], n: int) -> float:
    """Fraction of n-gram positions that are repeats (global within the sequence)."""
    if len(tokens) < n:
        return 0.0
    counts = Counter(ngrams(tokens, n))
    total_positions = len(tokens) - n + 1
    repeated_positions = sum(c - 1 for c in counts.values() if c > 1)
    return repeated_positions / max(1, total_positions)


def longest_common_prefix_len(a: List[str], b: List[str]) -> int:
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


@dataclass
class TextStats:
    chars: int
    words: int
    toks: int


def compute_text_stats(text: str) -> TextStats:
    toks = simple_tokens(text)
    words = sum(1 for t in toks if re.match(r"^[A-Za-z0-9]", t))
    return TextStats(chars=len(text), words=words, toks=len(toks))


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                yield {"__parse_error__": True, "__line_no__": line_no, "__error__": str(e), "__raw__": line}


def safe_get_str(obj: Dict[str, Any], key: str) -> str:
    v = obj.get(key, "")
    if v is None:
        return ""
    if not isinstance(v, str):
        return str(v)
    return v


def agg_int(xs: List[int]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(statistics.mean(xs)),
        "median": float(statistics.median(xs)),
        "p90": float(percentile([float(x) for x in xs], 90)),
        "p95": float(percentile([float(x) for x in xs], 95)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def agg_float(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(statistics.mean(xs)),
        "median": float(statistics.median(xs)),
        "p90": float(percentile(xs, 90)),
        "p95": float(percentile(xs, 95)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


# ----------------------------
# Example sampling (reservoir)
# ----------------------------
def reservoir_sample_jsonl(path: Path, k: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Uniform random k samples from a JSONL file without loading it all."""
    sample: List[Dict[str, Any]] = []
    seen = 0
    for ex in read_jsonl(path):
        if ex.get("__parse_error__"):
            continue
        seen += 1
        if len(sample) < k:
            sample.append(ex)
        else:
            j = rng.randrange(seen)
            if j < k:
                sample[j] = ex
        if MAX_SCAN_EXAMPLES_PER_FILE is not None and seen >= MAX_SCAN_EXAMPLES_PER_FILE:
            break
    return sample


# ----------------------------
# Per-file analysis
# ----------------------------
def analyze_one_file(path: Path) -> Dict[str, Any]:
    prompt_chars: List[int] = []
    prompt_words: List[int] = []
    prompt_toks: List[int] = []

    resp_chars: List[int] = []
    resp_words: List[int] = []
    resp_toks: List[int] = []

    vocab_prompt = Counter()
    vocab_resp = Counter()

    resp_unigrams_total = 0
    resp_unigrams_unique = set()
    resp_bigrams_total = 0
    resp_bigrams_unique = set()

    rep4_rates: List[float] = []

    role_artifact = 0
    system_prompt_leak = 0
    empty_resp = 0
    prompt_echo_lcp = 0
    prompt_in_resp_substring = 0
    parse_errors = 0

    source_counter = Counter()

    it = read_jsonl(path)
    if tqdm is not None:
        it = tqdm(it, desc=f"{path.name}", unit="ex", leave=False)

    n = 0
    for ex in it:
        if ex.get("__parse_error__"):
            parse_errors += 1
            continue

        n += 1
        if MAX_SCAN_EXAMPLES_PER_FILE is not None and n > MAX_SCAN_EXAMPLES_PER_FILE:
            break

        prompt = safe_get_str(ex, "prompt")
        response = safe_get_str(ex, "response")
        source = safe_get_str(ex, "source") or "unknown"
        source_counter[source] += 1

        if not response.strip():
            empty_resp += 1

        ps = compute_text_stats(prompt)
        rs = compute_text_stats(response)

        prompt_chars.append(ps.chars)
        prompt_words.append(ps.words)
        prompt_toks.append(ps.toks)

        resp_chars.append(rs.chars)
        resp_words.append(rs.words)
        resp_toks.append(rs.toks)

        p_toks = [t.lower() for t in simple_tokens(prompt)]
        r_toks = [t.lower() for t in simple_tokens(response)]

        vocab_prompt.update(p_toks)
        vocab_resp.update(r_toks)

        resp_unigrams_total += len(r_toks)
        resp_unigrams_unique.update(r_toks)

        bigs = list(ngrams(r_toks, 2))
        resp_bigrams_total += len(bigs)
        resp_bigrams_unique.update(bigs)

        rep4_rates.append(repeated_ngram_rate(r_toks, 4))

        if ROLE_ARTIFACT_RE.search(response):
            role_artifact += 1
        if SYSTEM_PROMPT_SNIPPET_RE.search(response):
            system_prompt_leak += 1

        # prompt echo heuristics
        if p_toks and r_toks:
            lcp = longest_common_prefix_len(p_toks, r_toks)
            if lcp >= 10 and lcp >= int(0.15 * len(p_toks)):
                prompt_echo_lcp += 1

        if prompt.strip() and prompt.strip() in response:
            prompt_in_resp_substring += 1

    if n == 0 and parse_errors > 0:
        return {
            "file": str(path),
            "n_examples": 0,
            "parse_errors": parse_errors,
            "error": "All lines failed JSON parsing.",
        }

    distinct_1 = (len(resp_unigrams_unique) / resp_unigrams_total) if resp_unigrams_total else 0.0
    distinct_2 = (len(resp_bigrams_unique) / resp_bigrams_total) if resp_bigrams_total else 0.0

    total_resp_tokens = sum(vocab_resp.values())
    vocab_size_resp = len(vocab_resp)
    ttr_resp = (vocab_size_resp / total_resp_tokens) if total_resp_tokens else 0.0

    total_prompt_tokens = sum(vocab_prompt.values())
    vocab_size_prompt = len(vocab_prompt)
    ttr_prompt = (vocab_size_prompt / total_prompt_tokens) if total_prompt_tokens else 0.0

    return {
        "file": str(path),
        "n_examples": n,
        "parse_errors": parse_errors,
        "source_distribution": dict(source_counter),
        "prompt_length": {
            "chars": agg_int(prompt_chars),
            "words": agg_int(prompt_words),
            "tokens": agg_int(prompt_toks),
        },
        "response_length": {
            "chars": agg_int(resp_chars),
            "words": agg_int(resp_words),
            "tokens": agg_int(resp_toks),
        },
        "vocab": {
            "prompt": {
                "vocab_size": vocab_size_prompt,
                "total_tokens": total_prompt_tokens,
                "type_token_ratio": ttr_prompt,
                "top_30_tokens": vocab_prompt.most_common(30),
            },
            "response": {
                "vocab_size": vocab_size_resp,
                "total_tokens": total_resp_tokens,
                "type_token_ratio": ttr_resp,
                "top_30_tokens": vocab_resp.most_common(30),
            },
        },
        "diversity": {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "repetition_4gram": agg_float(rep4_rates),
        },
        "artifacts": {
            "empty_response": {"count": empty_resp, "rate": empty_resp / max(1, n)},
            "role_prefix_leak": {"count": role_artifact, "rate": role_artifact / max(1, n)},
            "system_prompt_leak": {"count": system_prompt_leak, "rate": system_prompt_leak / max(1, n)},
            "prompt_echo_lcp": {"count": prompt_echo_lcp, "rate": prompt_echo_lcp / max(1, n)},
            "prompt_in_response_substring": {"count": prompt_in_resp_substring, "rate": prompt_in_resp_substring / max(1, n)},
        },
    }


# ----------------------------
# Folder scan + combined report
# ----------------------------
def find_synth_dir() -> Path:
    for cand in SYNTH_DIR_CANDIDATES:
        if cand.exists() and cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Could not find synth folder. Tried:\n  " + "\n  ".join(str(p) for p in SYNTH_DIR_CANDIDATES)
    )


def main() -> None:
    synth_dir = find_synth_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in synth_dir.glob("*.jsonl") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in: {synth_dir}")

    rng = random.Random(RNG_SEED)

    print(f"Scanning synth folder: {synth_dir}")
    print(f"Found {len(files)} jsonl files.")
    print(f"Sampling {EXAMPLES_PER_FILE} examples per file -> {EXAMPLES_TXT}")

    per_file_reports: List[Dict[str, Any]] = []
    examples_blocks: List[str] = []

    iterator = files
    if tqdm is not None:
        iterator = tqdm(files, desc="Files", unit="file")

    for f in iterator:
        # stats
        rep = analyze_one_file(f)
        per_file_reports.append(rep)

        # samples
        samples = reservoir_sample_jsonl(f, EXAMPLES_PER_FILE, rng)
        examples_blocks.append("=" * 88)
        examples_blocks.append(f"FILE: {f.name}")
        examples_blocks.append("=" * 88)

        if not samples:
            examples_blocks.append("(No valid JSON examples parsed.)\n")
            continue

        for i, ex in enumerate(samples, start=1):
            prompt = safe_get_str(ex, "prompt")
            response = safe_get_str(ex, "response")
            source = safe_get_str(ex, "source") or "unknown"
            examples_blocks.append(f"\n--- Example {i} (source={source}) ---")
            examples_blocks.append("PROMPT:")
            examples_blocks.append(prompt.strip())
            examples_blocks.append("\nRESPONSE:")
            examples_blocks.append(response.strip())
            examples_blocks.append("")  # spacer

        examples_blocks.append("")  # spacer after each file

    # Combined report (just stitching per-file; global aggregation optional but not requested)
    combined_report: Dict[str, Any] = {
        "synth_dir": str(synth_dir),
        "n_files": len(files),
        "files": [f.name for f in files],
        "per_file": per_file_reports,
        "notes": {
            "tokenizer": "simple regex tokenizer (words + punctuation)",
            "examples_sampling": f"{EXAMPLES_PER_FILE} reservoir samples per file, seed={RNG_SEED}",
            "max_scan_examples_per_file": MAX_SCAN_EXAMPLES_PER_FILE,
        },
    }

    json_path = OUT_DIR / f"{REPORT_STEM}.json"
    txt_path = OUT_DIR / f"{REPORT_STEM}.txt"

    json_path.write_text(json.dumps(combined_report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Small readable summary table
    lines: List[str] = []
    lines.append(f"Synth folder: {synth_dir}")
    lines.append(f"Files: {len(files)}")
    lines.append("")
    lines.append("Per-file quick table:")
    lines.append("  file\tN\tresp_tok_mean\td1\td2\trep4_mean\trole_leak\tsys_leak\tprompt_echo")
    for rep in per_file_reports:
        name = Path(rep["file"]).name
        if rep.get("n_examples", 0) <= 0:
            lines.append(f"  {name}\t0\t-\t-\t-\t-\t-\t-\t-")
            continue
        rt_mean = rep["response_length"]["tokens"]["mean"]
        d1 = rep["diversity"]["distinct_1"]
        d2 = rep["diversity"]["distinct_2"]
        rep4m = rep["diversity"]["repetition_4gram"]["mean"]
        rl = rep["artifacts"]["role_prefix_leak"]["rate"]
        sl = rep["artifacts"]["system_prompt_leak"]["rate"]
        pe = rep["artifacts"]["prompt_echo_lcp"]["rate"]
        lines.append(f"  {name}\t{rep['n_examples']}\t{rt_mean:.2f}\t{d1:.3f}\t{d2:.3f}\t{rep4m:.4f}\t{rl:.3%}\t{sl:.3%}\t{pe:.3%}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    EXAMPLES_TXT.write_text("\n".join(examples_blocks), encoding="utf-8")

    print("\n=== Done ===")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {EXAMPLES_TXT}")


if __name__ == "__main__":
    main()
