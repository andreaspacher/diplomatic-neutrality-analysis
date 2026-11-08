#!/usr/bin/env python3
"""
01_extract_mentions.py

Scans Data/*.jsonl (GlobalDiplomacyNet) for sentence-level country-to-country
mentions.

Direction is inferred from the file name (<ISO3>_<exec|mofa><n>.jsonl):
  - file country in SOURCE_COUNTRIES    -> direction "given"
        what a source country says about RUS / UKR
  - file country in RECEIVING_COUNTRIES -> direction "received"
        what RUS / UKR say about the source countries

Matching is alias-based (Data/country_aliases.json), case-insensitive,
whole-word, longest alias first (a single combined regex per direction, so
"Russian Federation" wins over "Russia" and spans never overlap).

One output row per mention, carrying the containing sentence. Downstream
(02_score_absa.py) scores unique (sentence_text, aspect) pairs.

Outputs (Output/):
  01_mentions.parquet      one row per mention
  01_alias_counts.csv      how often each alias fired (audit for noisy aliases)
  01_summary.txt           counts per direction / source / target, skipped files

Usage:
  python Scripts/01_extract_mentions.py
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import time
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
from config import ALIASES_JSON, DATA_DIR, OUTPUT_DIR, RECEIVING_COUNTRIES, SOURCE_COUNTRIES

FNAME_PATTERN = re.compile(r"^([A-Z]{3})_(exec|mofa)(\d*)\.jsonl$")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_sentence_tokenizer():
    """Punkt tokenizer exposing .span_tokenize(); works with old and new NLTK."""
    try:
        from nltk.tokenize.punkt import PunktTokenizer  # nltk >= 3.9
        try:
            return PunktTokenizer("english")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return PunktTokenizer("english")
    except ImportError:
        try:
            return nltk.data.load("tokenizers/punkt/english.pickle")
        except LookupError:
            nltk.download("punkt", quiet=True)
            return nltk.data.load("tokenizers/punkt/english.pickle")


def build_matcher(alias_dict: dict, isos: list[str]):
    """
    Returns (compiled_regex, lookup) where lookup maps casefolded alias -> (iso3, alias).
    Alternatives are ordered longest-first so the longest alias at a given
    position wins. Lookarounds instead of \\b so aliases ending in non-word
    characters (e.g. "Bharat (India)") still match.
    """
    lookup: dict[str, tuple[str, str]] = {}
    for iso in isos:
        for alias in alias_dict.get(iso, []):
            key = alias.casefold()
            if key in lookup and lookup[key][0] != iso:
                print(f"WARNING: alias {alias!r} belongs to both {lookup[key][0]} and {iso}; "
                      f"keeping {lookup[key][0]}.")
                continue
            lookup.setdefault(key, (iso, alias))
    if not lookup:
        raise SystemExit(f"No aliases found for {isos} in alias dictionary.")
    ordered = sorted(lookup, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(a) for a in ordered) + r")(?!\w)",
        re.IGNORECASE,
    )
    return pattern, lookup


def clean(s: str) -> str:
    return " ".join(s.split())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--aliases", type=Path, default=ALIASES_JSON)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.aliases, "r", encoding="utf-8") as f:
        alias_dict = json.load(f)

    matchers = {
        "given": build_matcher(alias_dict, RECEIVING_COUNTRIES),
        "received": build_matcher(alias_dict, SOURCE_COUNTRIES),
        "dyadic_UKR": build_matcher(alias_dict, ["UKR"]),   # RUS files: what Russia says about Ukraine
        "dyadic_RUS": build_matcher(alias_dict, ["RUS"]),   # UKR files: what Ukraine says about Russia
    }
    sent_tok = load_sentence_tokenizer()

    files = sorted(args.data_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files in {args.data_dir}")

    rows: list[dict] = []
    alias_counts: Counter = Counter()
    skipped: list[tuple[str, str]] = []
    n_docs = n_docs_hit = n_bad_json = n_unresolved = 0

    for path in tqdm(files, desc="files"):
        m = FNAME_PATTERN.match(path.name)
        if not m:
            skipped.append((path.name, "file name does not match <ISO3>_<exec|mofa><n>.jsonl"))
            continue
        iso3, source_type, _ = m.groups()

        if iso3 in SOURCE_COUNTRIES:
            scans = [("given", *matchers["given"])]
        elif iso3 in RECEIVING_COUNTRIES:
            other = "UKR" if iso3 == "RUS" else "RUS"
            scans = [("received", *matchers["received"]),
                     ("dyadic", *matchers[f"dyadic_{other}"])]
        else:
            skipped.append((path.name, f"{iso3} not in scope"))
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_bad_json += 1
                    continue

                text = rec.get("content") or ""
                if not text.strip():
                    continue
                n_docs += 1

                found = [(direction, lookup, mt) for direction, pattern, lookup in scans
                         for mt in pattern.finditer(text)]
                if not found:
                    continue

                spans = list(sent_tok.span_tokenize(text))
                starts = [s for s, _ in spans]
                doc_id = rec.get("id") or f"{path.name}:{line_no}"
                hit = False

                for direction, lookup, mt in found:
                    hit_key = lookup.get(mt.group(0).casefold())
                    if hit_key is None:  # exotic casefolding edge case
                        n_unresolved += 1
                        continue
                    target_iso3, alias = hit_key
                    hit = True

                    idx = max(bisect.bisect_right(starts, mt.start()) - 1, 0)
                    s, e = spans[idx]
                    rows.append({
                        "doc_id": doc_id,
                        "file": path.name,
                        "url": rec.get("url"),
                        "date": rec.get("date"),
                        "source_iso3": iso3,
                        "source_type": source_type,
                        "direction": direction,
                        "target_iso3": target_iso3,
                        "matched_alias": alias,
                        "matched_text": mt.group(0),
                        "char_start": mt.start(),
                        "char_end": mt.end(),
                        "sentence_idx": idx,
                        "sentence_text": clean(text[s:e]),
                    })
                    alias_counts[(direction, target_iso3, alias)] += 1
                n_docs_hit += hit

    if not rows:
        raise SystemExit("No mentions found. Check alias dictionary and file naming.")

    df = pd.DataFrame(rows)
    df.to_parquet(args.out_dir / "01_mentions.parquet", index=False)

    ac = (pd.DataFrame([(*k, v) for k, v in alias_counts.items()],
                       columns=["direction", "target_iso3", "alias", "n"])
          .sort_values(["direction", "target_iso3", "n"], ascending=[True, True, False]))
    ac.to_csv(args.out_dir / "01_alias_counts.csv", index=False)

    lines = [
        f"01_extract_mentions  ({time.strftime('%Y-%m-%d %H:%M:%S')}, {time.time() - t0:.0f}s)",
        f"files scanned: {len(files) - len(skipped)} / {len(files)}",
        f"documents with text: {n_docs}   with >=1 mention: {n_docs_hit}",
        f"mentions: {len(df)}   unique sentences: {df[['doc_id', 'sentence_idx']].drop_duplicates().shape[0]}",
        f"malformed JSON lines: {n_bad_json}   unresolved matches: {n_unresolved}",
        "",
        "Mentions by direction / source / target:",
        df.groupby(["direction", "source_iso3", "target_iso3"]).size().to_string(),
        "",
        "Skipped files:",
        *(f"  {n}: {why}" for n, why in skipped),
    ]
    summary = "\n".join(lines)
    (args.out_dir / "01_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()