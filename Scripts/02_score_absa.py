#!/usr/bin/env python3
"""
02_score_absa.py

Aspect-based sentiment analysis (ABSA) of every sentence in which a
country-to-country mention occurs, i.e. sentiment TOWARD the mentioned country
(not of the sentence as a whole).

Input : Output/01_mentions.parquet
Model : yangheng/deberta-v3-base-absa-v1.1 (Hugging Face; ~1 GB download on first run)
Unit  : unique (sentence_text, aspect_term) pair; aspect_term = the literal
        string matched in the sentence (column `matched_text`, e.g. "Moscow").

Outputs (Output/):
  02_absa_checkpoint.jsonl   append-only scoring checkpoint (resumed automatically)
  02_mentions_absa.parquet   01_mentions + absa_label, absa_prob_*, absa_score
  02_summary.txt             label distribution, timing

absa_score = P(positive) - P(negative), in [-1, 1].

Usage:
  python Scripts/02_score_absa.py --limit 200   # timing test (no merge)
  python Scripts/02_score_absa.py               # full run; re-run to resume
  python Scripts/02_score_absa.py --fresh       # discard checkpoint, start over

Long run? Use tmux/nohup. Requirements: pip install torch transformers sentencepiece
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:
    raise SystemExit("Run: pip install torch transformers sentencepiece") from e

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
from config import OUTPUT_DIR
INPUT_PARQUET = OUTPUT_DIR / "01_mentions.parquet"
CHECKPOINT = OUTPUT_DIR / "02_absa_checkpoint.jsonl"
OUT_PARQUET = OUTPUT_DIR / "02_mentions_absa.parquet"
OUT_SUMMARY = OUTPUT_DIR / "02_summary.txt"

MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device} ...")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    label2id = {v.lower(): k for k, v in model.config.id2label.items()}
    missing = {"positive", "negative", "neutral"} - set(label2id)
    if missing:
        raise ValueError(f"Model labels {model.config.id2label} lack {missing}")
    print(f"Model labels: {model.config.id2label}")
    return tok, model, device, model.config.id2label, label2id


@torch.inference_mode()
def score_batch(tok, model, device, id2label, label2id, texts, aspects, max_length):
    enc = tok(texts, aspects, padding=True, truncation=True,
              max_length=max_length, return_tensors="pt").to(device)
    probs = torch.softmax(model(**enc).logits, dim=-1).cpu()
    top = probs.argmax(dim=-1)
    out = []
    for i in range(len(texts)):
        out.append({
            "absa_label": id2label[top[i].item()],
            "absa_prob_positive": round(probs[i, label2id["positive"]].item(), 5),
            "absa_prob_negative": round(probs[i, label2id["negative"]].item(), 5),
            "absa_prob_neutral": round(probs[i, label2id["neutral"]].item(), 5),
        })
    return out


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #
def pair_key(sentence: str, aspect: str) -> str:
    return hashlib.md5(f"{sentence}\x1f{aspect}".encode("utf-8")).hexdigest()


def load_checkpoint_keys() -> set[str]:
    keys: set[str] = set()
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    keys.add(json.loads(line)["pair_key"])
                except (json.JSONDecodeError, KeyError):
                    continue  # partial last line after a crash
    return keys


def ensure_trailing_newline(path: Path):
    """After a hard kill the last line may be truncated; don't glue new records onto it."""
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=INPUT_PARQUET)
    ap.add_argument("--aspect-col", default="matched_text",
                    help="column used as aspect term (default: matched_text; alternative: matched_alias)")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="score only N pairs (timing test, no merge)")
    ap.add_argument("--fresh", action="store_true", help="delete the checkpoint and start over")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.input.exists():
        raise SystemExit(f"{args.input} not found. Run 01_extract_mentions.py first.")

    df = pd.read_parquet(args.input)
    df["sentence_text"] = df["sentence_text"].fillna("")
    df["aspect_term"] = df[args.aspect_col].fillna("")
    df["pair_key"] = [pair_key(s, a) for s, a in zip(df["sentence_text"], df["aspect_term"])]

    unique = (df[["pair_key", "sentence_text", "aspect_term"]]
              .drop_duplicates("pair_key").reset_index(drop=True))
    print(f"{len(df)} mentions -> {len(unique)} unique (sentence, aspect) pairs")

    if args.fresh and CHECKPOINT.exists():
        CHECKPOINT.unlink()
    done = load_checkpoint_keys()
    todo = unique[~unique["pair_key"].isin(done)].copy()
    print(f"{len(done)} already in checkpoint, {len(todo)} to score")

    if args.limit:
        todo = todo.head(args.limit)
        print(f"--limit: scoring {len(todo)} pairs")

    # sort by length -> minimal padding per batch -> much faster
    todo = todo.assign(_len=todo["sentence_text"].str.len()).sort_values("_len").drop(columns="_len")

    t0 = time.time()
    n_scored = 0
    if len(todo):
        tok, model, device, id2label, label2id = load_model(args.model)
        ensure_trailing_newline(CHECKPOINT)
        with open(CHECKPOINT, "a", encoding="utf-8") as ck:
            for i in tqdm(range(0, len(todo), args.batch_size), desc="ABSA batches"):
                chunk = todo.iloc[i:i + args.batch_size]
                res = score_batch(tok, model, device, id2label, label2id,
                                  chunk["sentence_text"].tolist(),
                                  chunk["aspect_term"].tolist(), args.max_length)
                for key, r in zip(chunk["pair_key"], res):
                    ck.write(json.dumps({"pair_key": key, **r}) + "\n")
                ck.flush()
                n_scored += len(chunk)
    elapsed = time.time() - t0

    if args.limit:
        rate = n_scored / elapsed if elapsed else float("nan")
        remaining = len(unique) - len(done) - n_scored
        print(f"\nTiming test: {n_scored} pairs in {elapsed:.0f}s ({rate:.1f} pairs/s). "
              f"Estimated time for the remaining {remaining}: {remaining / rate / 3600:.1f} h")
        return

    # ---- merge checkpoint back onto the full mention table --------------- #
    ck_df = pd.read_json(CHECKPOINT, lines=True).drop_duplicates("pair_key", keep="last")
    merged = df.merge(ck_df, on="pair_key", how="left")
    merged["absa_score"] = merged["absa_prob_positive"] - merged["absa_prob_negative"]
    merged.to_parquet(OUT_PARQUET, index=False)

    n_missing = merged["absa_label"].isna().sum()
    lines = [
        f"02_score_absa  ({time.strftime('%Y-%m-%d %H:%M:%S')})",
        f"model: {args.model}   aspect column: {args.aspect_col}   max_length: {args.max_length}",
        f"mentions: {len(merged)}   unique pairs: {len(unique)}   scored this run: {n_scored} ({elapsed / 60:.1f} min)",
        f"unscored mentions: {n_missing}",
        "",
        "Label distribution (share) by direction / target:",
        (merged.groupby(["direction", "target_iso3"])["absa_label"]
               .value_counts(normalize=True).round(3).unstack().to_string()),
        "",
        "Mean absa_score by direction / target:",
        merged.groupby(["direction", "target_iso3"])["absa_score"].mean().round(3).to_string(),
    ]
    summary = "\n".join(lines)
    OUT_SUMMARY.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nWrote {OUT_PARQUET}")


if __name__ == "__main__":
    main()