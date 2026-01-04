from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ============================================================
# Utilities
# ============================================================

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Text cleaning for synthetic generations / eval generations
# ============================================================

_ROLE_SPLITS = [
    "\nassistant\n", "\nassistant:", "\nuser\n", "\nuser:", "\nsystem\n", "\nsystem:",
    "<|assistant|>", "<|user|>", "<|system|>",
]

def clean_generated_response(text: str) -> str:
    """Remove common chat-template echoes and keep only the assistant\'s first answer."""
    if text is None:
        return ""
    t = str(text).strip()

    # If the model echoed a transcript, keep only the last assistant segment.
    for m in ["\nassistant\n", "\nassistant:", "<|assistant|>", "assistant\n", "assistant:"]:
        if m in t:
            t = t.split(m)[-1].strip()

    # Drop typical role-only lines and the exact system prompt line if it appears.
    lines = [ln.strip() for ln in t.splitlines()]
    # Remove leading empties
    while lines and lines[0] == "":
        lines.pop(0)
    # Remove leading role tags like "system", "user", "assistant"
    while lines and lines[0].lower() in ("system", "user", "assistant"):
        lines.pop(0)
        while lines and lines[0] == "":
            lines.pop(0)
    # Remove the common default system prompt if it got echoed
    if lines and lines[0].lower().startswith("you are a helpful conversational agent"):
        lines.pop(0)
        while lines and lines[0] == "":
            lines.pop(0)

    t = "\n".join(lines).strip()

    # Remove leading role headers like "system:" / "user:" / "assistant:" if present
    t = re.sub(r"^\s*(system|user|assistant)\s*[:\n]+\s*", "", t, flags=re.IGNORECASE).strip()

    # Remove any remaining special role tokens
    t = t.replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", "").strip()

    # If the model starts generating another turn, cut at it.
    # This keeps only the assistant\'s first answer.
    cut_markers = ["\nUser:", "\nUSER:", "\nuser:", "\nAssistant:", "\nASSISTANT:"]
    cut_pos = min([t.find(m) for m in cut_markers if m in t] or [-1])
    if cut_pos != -1:
        t = t[:cut_pos].strip()

    # Remove obvious duplicated "User:"/"Assistant:" prefixes
    t = re.sub(r"^\s*(User|Assistant)\s*:\s*", "", t).strip()

    return t

# ============================================================
# Dataset building: conversations -> (prompt, response) pairs
# ============================================================

def _normalize_speaker(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"user", "human"}:
        return "user"
    if s in {"system", "assistant", "bot"}:
        return "system"
    # Fallback: treat unknown as user to avoid losing data
    return "user"

def load_conversations(source_json: Path) -> List[Dict[str, Any]]:
    """
    Accepts a JSON file that is a list of conversations.
    Each conversation should have a 'turns' field:
      turns: [{speaker: "user"/"system", utterance: "..."} ...]
    Extra fields are ignored.
    """
    data = read_json(source_json)
    if not isinstance(data, list):
        raise ValueError("source_json must be a JSON list of conversations.")
    convs: List[Dict[str, Any]] = []
    for i, c in enumerate(data):
        turns = c.get("turns") or c.get("dialogue") or c.get("messages")
        if not isinstance(turns, list):
            continue
        norm_turns = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            sp = _normalize_speaker(t.get("speaker", ""))
            utt = t.get("utterance", "")
            if utt is None:
                utt = ""
            utt = str(utt).strip()
            if utt == "":
                continue
            norm_turns.append({"speaker": sp, "utterance": utt})
        if len(norm_turns) < 2:
            continue
        convs.append({"conv_id": c.get("dialogue_id") or c.get("conv_id") or f"conv-{i}", "turns": norm_turns})
    if not convs:
        raise ValueError("No usable conversations found in source_json.")
    return convs

def build_pairs_from_conversations(
    convs: List[Dict[str, Any]],
    max_context_turns: int,
) -> List[Dict[str, str]]:
    """
    Extract (prompt, response) where a user turn is immediately followed by a system turn.
    prompt ends with 'Assistant:' so generation continues cleanly.
    """
    pairs: List[Dict[str, str]] = []
    for c in convs:
        turns = c["turns"]
        for j in range(len(turns) - 1):
            t0 = turns[j]
            t1 = turns[j + 1]
            if t0["speaker"] != "user" or t1["speaker"] != "system":
                continue
            # context = up to max_context_turns turns before t1 (excluding t1)
            start = max(0, (j + 1) - max_context_turns)
            ctx = turns[start : j + 1]
            lines = []
            for t in ctx:
                role = "User" if t["speaker"] == "user" else "Assistant"
                lines.append(f"{role}: {t['utterance']}")
            lines.append("Assistant:")  # generation prompt
            prompt = "\n".join(lines)
            response = t1["utterance"]
            pairs.append({"prompt": prompt, "response": response})
    return pairs

# ============================================================
# Prepare command (runner-compatible)
# ============================================================

def prepare_dataset(
    out_dir: Path,
    source_json: Path,
    seed: int,
    generations: int,
    train_pairs_per_gen: int,
    test_pairs: int,
    anchor_prompts: int,
    max_context_turns: int,
    val_frac: float,
    test_frac: float,
) -> None:
    """
    Creates:
      data/real_train_gen00.jsonl
      data/real_train_gen01.jsonl .. gen{G}.jsonl
      data/train_prompts_gen01.jsonl .. gen{G}.jsonl
      data/real_test.jsonl
      data/anchor_prompts.jsonl
      data/prepare_meta.json
    Splits are created by shuffling conversations (not original dataset splits).
    """
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    convs = load_conversations(source_json)
    rng = random.Random(seed)
    rng.shuffle(convs)

    n = len(convs)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_test + n_val >= n:
        raise ValueError("val_frac + test_frac too large for number of conversations.")

    test_convs = convs[:n_test]
    val_convs = convs[n_test:n_test + n_val]  # currently unused but reserved
    train_convs = convs[n_test + n_val :]

    # Build pairs
    train_pairs_all = build_pairs_from_conversations(train_convs, max_context_turns=max_context_turns)
    test_pairs_all = build_pairs_from_conversations(test_convs, max_context_turns=max_context_turns)

    rng.shuffle(train_pairs_all)
    rng.shuffle(test_pairs_all)

    # real_test
    real_test = test_pairs_all[: min(test_pairs, len(test_pairs_all))]
    write_jsonl(out_dir / "real_test.jsonl", [{"prompt": r["prompt"], "response": r["response"], "source": "real"} for r in real_test])

    # anchor prompts sampled from test prompts (fallback to train if needed)
    anchor_pool = [r["prompt"] for r in real_test]
    if len(anchor_pool) < anchor_prompts:
        anchor_pool = [r["prompt"] for r in test_pairs_all] + [r["prompt"] for r in train_pairs_all]
    anchor_pool = list(dict.fromkeys(anchor_pool))  # unique while preserving order
    rng.shuffle(anchor_pool)
    anchor = anchor_pool[: min(anchor_prompts, len(anchor_pool))]
    write_jsonl(out_dir / "anchor_prompts.jsonl", [{"prompt": p} for p in anchor])

    # Allocate disjoint real shards: gen00..genG
    total_needed = (generations + 1) * train_pairs_per_gen
    if len(train_pairs_all) < total_needed:
        raise ValueError(
            f"Not enough training pairs for requested shards. "
            f"Have {len(train_pairs_all)}, need {total_needed}."
        )

    idx = 0
    for g in range(0, generations + 1):
        shard = train_pairs_all[idx : idx + train_pairs_per_gen]
        idx += train_pairs_per_gen
        out_path = out_dir / f"real_train_gen{g:02d}.jsonl"
        write_jsonl(out_path, [{"prompt": r["prompt"], "response": r["response"], "source": "real"} for r in shard])

        # prompts for synthesis start at gen01..genG
        if g >= 1:
            p_out = out_dir / f"train_prompts_gen{g:02d}.jsonl"
            write_jsonl(p_out, [{"prompt": r["prompt"]} for r in shard])

    meta = {
        "seed": seed,
        "source_json": str(source_json),
        "n_conversations": n,
        "n_train_conversations": len(train_convs),
        "n_val_conversations": len(val_convs),
        "n_test_conversations": len(test_convs),
        "max_context_turns": max_context_turns,
        "train_pairs_total": len(train_pairs_all),
        "test_pairs_total": len(test_pairs_all),
        "train_pairs_per_gen": train_pairs_per_gen,
        "generations": generations,
        "test_pairs_written": len(real_test),
        "anchor_prompts_written": len(anchor),
        "shards_written": [f"real_train_gen{g:02d}.jsonl" for g in range(generations + 1)],
    }
    write_json(out_dir / "prepare_meta.json", meta)
    print("Prepared data in:", out_dir)

# ============================================================
# Training
# ============================================================

@dataclass
class CausalExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]

class JsonlCausalDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_length: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        prompt = r["prompt"]
        response = r["response"]

        # Prompt and full text (prompt ends with Assistant:)
        prompt_text = prompt.rstrip() + " "
        full_text = prompt.rstrip() + " " + response.strip()

        p_ids = self.tok(prompt_text, add_special_tokens=False, truncation=True, max_length=self.max_length)["input_ids"]
        f = self.tok(full_text, add_special_tokens=False, truncation=True, max_length=self.max_length)
        f_ids = f["input_ids"]
        attn = f["attention_mask"]

        prompt_len = min(len(p_ids), len(f_ids))
        labels = [-100] * prompt_len + f_ids[prompt_len:]

        return {"input_ids": f_ids, "attention_mask": attn, "labels": labels}

def collate_fn(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attn, labels = [], [], []
    for x in batch:
        ids = x["input_ids"]
        am = x["attention_mask"]
        lab = x["labels"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append(am + [0] * pad)
        labels.append(lab + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

def train_lora(
    model_name: str,
    train_jsonl: Path,
    out_adapter: Path,
    epochs: float,
    batch: int,
    grad_accum: int,
    max_length: int,
    lr: float,
    seed: int,
) -> None:
    set_seed(seed)
    out_adapter.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Conservative LoRA defaults
    # Pick target modules that exist (Qwen/Llama-style by default).
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj"]
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in preferred if t in names]
    if not targets:
        # Some GPT-like models use different names.
        fallbacks = ["c_attn", "c_proj", "Wqkv", "wo"]
        targets = [t for t in fallbacks if t in names]
    if not targets:
        raise ValueError("Could not infer LoRA target_modules for this model. Set them manually in train_lora().")
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)

    rows = read_jsonl(train_jsonl)
    ds = JsonlCausalDataset(rows, tokenizer, max_length=max_length)

    args = TrainingArguments(
        output_dir=str(out_adapter / "_trainer_tmp"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_strategy="steps",
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        bf16=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    trainer.train()
    model.save_pretrained(out_adapter)
    tokenizer.save_pretrained(out_adapter)
    print("Saved adapter:", out_adapter)

# ============================================================
# Generate synthetic (batched), with cleaning
# ============================================================

@torch.no_grad()
def generate_synthetic(
    model_name: str,
    adapter_dir: Path,
    prompts_jsonl: Path,
    out_jsonl: Path,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    max_prompts: int,
) -> None:
    set_seed(seed)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    prompts_rows = read_jsonl(prompts_jsonl)
    prompts = [r["prompt"] for r in prompts_rows if r.get("prompt")]
    if max_prompts > 0:
        prompts = prompts[:max_prompts]

    rows_out: List[Dict[str, Any]] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating -> {out_jsonl.name}"):
        batch_prompts = prompts[i:i+batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        input_lens = enc["attention_mask"].sum(dim=1).tolist()
        for bp, seq, in_len in zip(batch_prompts, gen, input_lens):
            new_tokens = seq[int(in_len):]
            resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            resp = clean_generated_response(resp)
            rows_out.append({"prompt": bp, "response": resp, "source": "synthetic"})

    write_jsonl(out_jsonl, rows_out)
    print("Wrote synthetic:", out_jsonl)

# ============================================================
# Mix
# ============================================================

def mix_real_and_synth(
    real_train: Path,
    synth: Path,
    out_train: Path,
    total_n: int,
    synth_frac: float,
    seed: int,
) -> None:
    set_seed(seed)
    rng = random.Random(seed)

    real_rows = read_jsonl(real_train)
    synth_rows = read_jsonl(synth)

    n_synth = int(round(total_n * synth_frac))
    n_real = total_n - n_synth

    if n_real > len(real_rows):
        raise ValueError(f"Not enough real rows: need {n_real}, have {len(real_rows)}")
    if n_synth > len(synth_rows):
        raise ValueError(f"Not enough synth rows: need {n_synth}, have {len(synth_rows)}")

    rng.shuffle(real_rows)
    rng.shuffle(synth_rows)

    mixed = real_rows[:n_real] + synth_rows[:n_synth]
    rng.shuffle(mixed)

    # Drop empty responses defensively
    cleaned = []
    for r in mixed:
        if not r.get("response"):
            continue
        cleaned.append(r)

    write_jsonl(out_train, cleaned)
    print("Wrote mixed train:", out_train, f"(requested={total_n}, wrote={len(cleaned)}, synth_frac={synth_frac})")

# ============================================================
# Eval (PPL on real test + diversity on anchors), with cleaning
# ============================================================

@torch.no_grad()
def eval_perplexity(
    model_name: str,
    adapter_dir: Path,
    test_jsonl: Path,
    max_length: int,
    max_examples: int,
    batch_size: int = 8,
) -> float:
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    rows = read_jsonl(test_jsonl)
    if max_examples > 0:
        rows = rows[:max_examples]

    # Build tokenized examples with prompt-masked labels
    examples = []
    for r in rows:
        prompt = r["prompt"].rstrip() + " "
        full = r["prompt"].rstrip() + " " + r["response"].strip()

        p_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        f = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)
        f_ids = f["input_ids"]
        attn = f["attention_mask"]

        prompt_len = min(len(p_ids), len(f_ids))
        labels = [-100] * prompt_len + f_ids[prompt_len:]
        examples.append({"input_ids": f_ids, "attention_mask": attn, "labels": labels})

    def pad_batch(exs):
        max_len = max(len(x["input_ids"]) for x in exs)
        input_ids, attn, labels = [], [], []
        for x in exs:
            ids = x["input_ids"]
            am = x["attention_mask"]
            lab = x["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [tokenizer.pad_token_id]*pad)
            attn.append(am + [0]*pad)
            labels.append(lab + [-100]*pad)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    losses = []
    for i in tqdm(range(0, len(examples), batch_size), desc="Eval PPL"):
        b = examples[i:i+batch_size]
        input_ids, attention_mask, labels = pad_batch(b)
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        labels = labels.to(model.device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(float(out.loss.item()))

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")
    return ppl

def distinct_n(texts: List[str], n: int) -> float:
    if not texts:
        return 0.0
    total = 0
    uniq = set()
    for t in texts:
        toks = t.split()
        if len(toks) < n:
            continue
        total += max(0, len(toks) - n + 1)
        for i in range(len(toks) - n + 1):
            uniq.add(tuple(toks[i:i+n]))
    return (len(uniq) / total) if total > 0 else 0.0

def repetition_rate_4gram(texts: List[str]) -> float:
    reps = 0
    total = 0
    for t in texts:
        toks = t.split()
        if len(toks) < 4:
            continue
        grams = [tuple(toks[i:i+4]) for i in range(len(toks) - 3)]
        total += len(grams)
        reps += len(grams) - len(set(grams))
    return (reps / total) if total > 0 else 0.0

@torch.no_grad()
def eval_diversity_from_anchor(
    model_name: str,
    adapter_dir: Path,
    anchor_prompts_jsonl: Path,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    limit: int,
    batch_size: int = 16,
) -> Dict[str, float]:
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    prompts = [r["prompt"] for r in read_jsonl(anchor_prompts_jsonl) if r.get("prompt")]
    if limit > 0:
        prompts = prompts[:limit]

    outs: List[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Eval diversity"):
        batch_prompts = prompts[i:i+batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        input_lens = enc["attention_mask"].sum(dim=1).tolist()
        for seq, in_len in zip(gen, input_lens):
            new_tokens = seq[int(in_len):]
            resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            resp = clean_generated_response(resp)
            outs.append(resp)

    return {
        "distinct_1": distinct_n(outs, 1),
        "distinct_2": distinct_n(outs, 2),
        "repetition_4gram": repetition_rate_4gram(outs),
    }

def save_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    write_json(path, metrics)
    print("Wrote metrics:", path)

# ============================================================
# CLI (runner-compatible)
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--source_json", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--train_pairs_per_gen", type=int, default=10000)
    p.add_argument("--test_pairs", type=int, default=2000)
    p.add_argument("--anchor_prompts", type=int, default=300)
    p.add_argument("--max_context_turns", type=int, default=10)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--test_frac", type=float, default=0.05)

    t = sub.add_parser("train")
    t.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    t.add_argument("--train_jsonl", type=Path, required=True)
    t.add_argument("--out_adapter", type=Path, required=True)
    t.add_argument("--epochs", type=float, default=3.0)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--grad_accum", type=int, default=16)
    t.add_argument("--max_length", type=int, default=1024)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--seed", type=int, default=42)

    g = sub.add_parser("generate")
    g.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    g.add_argument("--adapter", type=Path, required=True)
    g.add_argument("--prompts_jsonl", type=Path, required=True)
    g.add_argument("--out_jsonl", type=Path, required=True)
    g.add_argument("--seed", type=int, default=123)
    g.add_argument("--max_new_tokens", type=int, default=100)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--top_p", type=float, default=0.95)
    g.add_argument("--batch_size", type=int, default=16)
    g.add_argument("--max_prompts", type=int, default=-1)

    m = sub.add_parser("mix")
    m.add_argument("--real_train", type=Path, required=True)
    m.add_argument("--synth", type=Path, required=True)
    m.add_argument("--out_train", type=Path, required=True)
    m.add_argument("--total_n", type=int, default=10000)
    m.add_argument("--synth_frac", type=float, required=True)
    m.add_argument("--seed", type=int, default=42)

    e = sub.add_parser("eval")
    e.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    e.add_argument("--adapter", type=Path, required=True)
    e.add_argument("--real_test", type=Path, required=True)
    e.add_argument("--anchor_prompts", type=Path, required=True)
    e.add_argument("--out_metrics", type=Path, required=True)
    e.add_argument("--max_length", type=int, default=1024)
    e.add_argument("--max_test_examples", type=int, default=800)
    e.add_argument("--anchor_limit", type=int, default=200)
    e.add_argument("--max_new_tokens", type=int, default=100)
    e.add_argument("--temperature", type=float, default=0.9)
    e.add_argument("--top_p", type=float, default=0.95)
    e.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if args.cmd == "prepare":
        prepare_dataset(
            out_dir=args.out,
            source_json=args.source_json,
            seed=args.seed,
            generations=args.generations,
            train_pairs_per_gen=args.train_pairs_per_gen,
            test_pairs=args.test_pairs,
            anchor_prompts=args.anchor_prompts,
            max_context_turns=args.max_context_turns,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )

    elif args.cmd == "train":
        train_lora(
            model_name=args.model,
            train_jsonl=args.train_jsonl,
            out_adapter=args.out_adapter,
            epochs=args.epochs,
            batch=args.batch,
            grad_accum=args.grad_accum,
            max_length=args.max_length,
            lr=args.lr,
            seed=args.seed,
        )

    elif args.cmd == "generate":
        generate_synthetic(
            model_name=args.model,
            adapter_dir=args.adapter,
            prompts_jsonl=args.prompts_jsonl,
            out_jsonl=args.out_jsonl,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            max_prompts=args.max_prompts,
        )

    elif args.cmd == "mix":
        mix_real_and_synth(
            real_train=args.real_train,
            synth=args.synth,
            out_train=args.out_train,
            total_n=args.total_n,
            synth_frac=args.synth_frac,
            seed=args.seed,
        )

    elif args.cmd == "eval":
        ppl = eval_perplexity(
            model_name=args.model,
            adapter_dir=args.adapter,
            test_jsonl=args.real_test,
            max_length=args.max_length,
            max_examples=args.max_test_examples,
        )
        div = eval_diversity_from_anchor(
            model_name=args.model,
            adapter_dir=args.adapter,
            anchor_prompts_jsonl=args.anchor_prompts,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            limit=args.anchor_limit,
        )
        metrics = {"perplexity_real_test": ppl, **div}
        save_metrics(args.out_metrics, metrics)

if __name__ == "__main__":
    main()
