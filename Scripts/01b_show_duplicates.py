#!/usr/bin/env python3
"""
01b_show_duplicates.py — which sentences does `03_build_panel.py --dedupe` collapse?

Mirrors the filters and the duplicate key of 03 (source country, target country, sentence
text; NOT the aspect term or the document) on Output/01_mentions.parquet.
Not mirrored: the "unscored" filter of 03 (irrelevant once 02 has finished).

Outputs (Output/), suffix _quarter with --by-quarter:
  01b_duplicates.csv         one row per duplicated sentence (n_mentions, n_removed, n_docs,
                             n_files, n_quarters, first/last/kept date, sentence), most repeated first
  01b_by_country.csv         mentions, removed, share removed per direction x source x target
  01b_removed_mentions.csv   every removed mention row (only with --write-removed)
  01b_summary.txt            totals, breakdowns, the 30 most repeated sentences

Options:
  --by-quarter   key includes the quarter (milder variant: each sentence counts once per quarter)
  --sort-date    keep the EARLIEST occurrence instead of the first in file order
  --min-count N  only list sentences with at least N mentions in 01b_duplicates.csv (default 2)

Usage: python Scripts/01b_show_duplicates.py
"""

from __future__ import annotations

import argparse
import textwrap

import pandas as pd

from config import OUTPUT_DIR, RECEIVING_COUNTRIES, SAMPLE_END, SAMPLE_START, SOURCE_COUNTRIES

INPUT = OUTPUT_DIR / "01_mentions.parquet"


def parse_dates(s: pd.Series) -> pd.Series:
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_localize(None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--by-quarter", action="store_true")
    ap.add_argument("--sort-date", action="store_true")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--write-removed", action="store_true")
    args = ap.parse_args()
    sfx = "_quarter" if args.by_quarter else ""

    if not INPUT.exists():
        raise SystemExit(f"{INPUT} not found. Run 01_extract_mentions.py first.")
    m = pd.read_parquet(INPUT)

    # ---- same filters as 03 -------------------------------------------- #
    m["dt"] = parse_dates(m["date"])
    m["quarter"] = m["dt"].dt.to_period("Q")
    q0, q1 = pd.Period(SAMPLE_START, "Q"), pd.Period(SAMPLE_END, "Q")
    m = m[m["quarter"].notna() & (m["quarter"] >= q0) & (m["quarter"] <= q1)]
    g = ((m["direction"] == "given") & m["source_iso3"].isin(SOURCE_COUNTRIES)
         & m["target_iso3"].isin(RECEIVING_COUNTRIES))
    r = ((m["direction"] == "received") & m["source_iso3"].isin(RECEIVING_COUNTRIES)
         & m["target_iso3"].isin(SOURCE_COUNTRIES))
    m = m[g | r].copy()
    if args.sort_date:
        m = m.sort_values("dt", kind="stable")
    m = m.reset_index(drop=True)
    m["quarter"] = m["quarter"].astype(str)

    key = ["direction", "source_iso3", "target_iso3"] + (["quarter"] if args.by_quarter else []) + ["sentence_text"]
    m["dup_rank"] = m.groupby(key, sort=False).cumcount()
    m["removed"] = m["dup_rank"] > 0

    # ---- one row per duplicated sentence ------------------------------- #
    grp = (m.groupby(key, sort=False)
             .agg(n_mentions=("dup_rank", "size"), n_docs=("doc_id", "nunique"), n_files=("file", "nunique"),
                  n_quarters=("quarter", "nunique"), first_date=("dt", "min"), last_date=("dt", "max"),
                  kept_date=("dt", "first"), kept_doc=("doc_id", "first"))
             .reset_index())
    grp["n_removed"] = grp["n_mentions"] - 1
    cols = ["direction", "source_iso3", "target_iso3", "n_mentions", "n_removed", "n_docs", "n_files",
            "n_quarters", "first_date", "last_date", "kept_date", "kept_doc", "sentence_text"]
    if args.by_quarter:
        cols.insert(3, "quarter")
    dups = (grp[grp["n_mentions"] >= args.min_count].sort_values("n_mentions", ascending=False)[cols])
    dups.to_csv(OUTPUT_DIR / f"01b_duplicates{sfx}.csv", index=False, encoding="utf-8-sig")

    # ---- breakdowns ----------------------------------------------------- #
    by = (m.groupby(["direction", "source_iso3", "target_iso3"])
            .agg(mentions=("removed", "size"), removed=("removed", "sum")).reset_index())
    by["share_removed"] = by["removed"] / by["mentions"]
    by["kept"] = by["mentions"] - by["removed"]
    by.to_csv(OUTPUT_DIR / f"01b_by_country{sfx}.csv", index=False)
    by_year = m[m["removed"]].groupby(m["dt"].dt.year).size().rename("removed")
    all_year = m.groupby(m["dt"].dt.year).size().rename("mentions")
    yr = pd.concat([all_year, by_year], axis=1).fillna(0).astype(int)
    yr["share_removed"] = (yr["removed"] / yr["mentions"]).round(3)

    n, n_rem = len(m), int(m["removed"].sum())
    multi = grp[grp["n_mentions"] > 1]
    lines = [
        f"01b_show_duplicates  key: {', '.join(key[:-1])}, sentence_text   "
        f"keep: {'earliest' if args.sort_date else 'first in file order'}",
        f"mentions in scope: {n}   kept: {n - n_rem}   removed: {n_rem} ({n_rem / n:.1%})",
        f"duplicated sentences: {len(multi)}   "
        f"spanning >1 document: {int((multi['n_docs'] > 1).sum())}   "
        f">1 quarter: {int((multi['n_quarters'] > 1).sum())}   >1 file: {int((multi['n_files'] > 1).sum())}",
        "", "Removed by direction x source x target:", by.round(3).to_string(index=False),
        "", "Removed by year:", yr.to_string(), "",
        "30 most repeated sentences (n_mentions | n_docs | n_quarters | src>tgt | sentence):",
    ]
    for _, x in dups.head(30).iterrows():
        lines.append(f"  {x.n_mentions:>5} | {x.n_docs:>4} | {x.n_quarters:>3} | {x.source_iso3}>{x.target_iso3} | "
                     + textwrap.shorten(x.sentence_text, 160, placeholder=" ..."))
    text = "\n".join(lines)
    (OUTPUT_DIR / f"01b_summary{sfx}.txt").write_text(text, encoding="utf-8")
    print(text)

    if args.write_removed:
        keep = ["doc_id", "file", "url", "date", "direction", "source_iso3", "target_iso3",
                "matched_text", "sentence_idx", "sentence_text"]
        m.loc[m["removed"], keep].to_csv(OUTPUT_DIR / f"01b_removed_mentions{sfx}.csv",
                                         index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()