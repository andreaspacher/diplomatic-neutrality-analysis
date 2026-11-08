#!/usr/bin/env python3
"""
03_build_panel.py

Aggregates mention-level ABSA scores (Output/02_mentions_absa.parquet) into a
country-quarter panel (one row per source country x quarter, full grid, so
missing cells appear explicitly as NaN).

Given side    (source country -> RUS / UKR):
  score_RUS, score_UKR   mean absa_score of mentions in the quarter
  n_RUS, n_UKR           number of mentions   (n_docs_*: number of distinct documents)
  tilt       = score_RUS - score_UKR      (positive = pro-Russian lean)
  asymmetry  = |tilt|
  valence    = (score_RUS + score_UKR) / 2
  balanced   = 1 if asymmetry < BALANCE_CUTOFF
  low_mention= 1 if min(n_RUS, n_UKR) < MIN_MENTIONS
Received side (RUS / UKR -> source country):
  recv_RUS, recv_UKR     mean score received from Russia / Ukraine
  recv_mean, recv_min    joint reception (mean / minimum of the two)
  recv_both_pos          1 if both > POSITIVE_THRESHOLD
Plus: group, year, quarter, t (0-based quarter index), post (>= invasion quarter).

Outputs (Output/):
  03_panel.parquet / 03_panel.csv   (suffix _dedup with --dedupe)
  03_summary.txt                    coverage and mention counts

Usage:
  python Scripts/03_build_panel.py
  python Scripts/03_build_panel.py --dedupe   # count each sentence once per (source, direction, target)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from config import (
    BALANCE_CUTOFF, GROUP_OF, INVASION_QUARTER, MIN_MENTIONS, OUTPUT_DIR,
    POSITIVE_THRESHOLD, RECEIVING_COUNTRIES, SAMPLE_END, SAMPLE_START,
    SOURCE_COUNTRIES,
)

INPUT = OUTPUT_DIR / "02_mentions_absa.parquet"


def parse_quarters(dates: pd.Series) -> pd.Series:
    try:
        dt = pd.to_datetime(dates, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):  # older pandas
        dt = pd.to_datetime(dates, errors="coerce", utc=True)
    return dt.dt.tz_localize(None).dt.to_period("Q")


def pivot_side(df, index_country, column_country, prefix_score, prefix_n, prefix_docs, quarters):
    """Aggregate to (country, quarter, other-country) and pivot the other country into columns."""
    agg = (df.groupby([index_country, "quarter", column_country])
             .agg(score=("absa_score", "mean"),
                  n=("absa_score", "size"),
                  n_docs=("doc_id", "nunique"))
             .reset_index())
    wide = agg.pivot(index=[index_country, "quarter"], columns=column_country,
                     values=["score", "n", "n_docs"])
    # flatten + rename
    wide.columns = [
        {"score": prefix_score, "n": prefix_n, "n_docs": prefix_docs}[a] + b
        for a, b in wide.columns
    ]
    for b in RECEIVING_COUNTRIES if column_country == "target_iso3" else RECEIVING_COUNTRIES:
        for p in (prefix_score, prefix_n, prefix_docs):
            if p + b not in wide.columns:
                wide[p + b] = np.nan
    wide = wide.reset_index().rename(columns={index_country: "country"})
    return wide


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path if False else str, default=str(INPUT))
    ap.add_argument("--dedupe", action="store_true",
                    help="count each sentence once per (source, direction, target)")
    args = ap.parse_args()
    suffix = "_dedup" if args.dedupe else ""
    t0 = time.time()
    report: list[str] = []

    df = pd.read_parquet(args.input)
    n0 = len(df)

    # ---- cleaning ------------------------------------------------------- #
    df["quarter"] = parse_quarters(df["date"])
    n_nat = df["quarter"].isna().sum()
    df = df[df["quarter"].notna()]

    q_start, q_end = pd.Period(SAMPLE_START, "Q"), pd.Period(SAMPLE_END, "Q")
    n_before_window = len(df)
    df = df[(df["quarter"] >= q_start) & (df["quarter"] <= q_end)]
    n_outside = n_before_window - len(df)

    n_unscored = df["absa_score"].isna().sum()
    df = df[df["absa_score"].notna()]

    # keep only the two valid directions
    given = df[(df["direction"] == "given") & df["source_iso3"].isin(SOURCE_COUNTRIES)
               & df["target_iso3"].isin(RECEIVING_COUNTRIES)]
    received = df[(df["direction"] == "received") & df["source_iso3"].isin(RECEIVING_COUNTRIES)
                  & df["target_iso3"].isin(SOURCE_COUNTRIES)]

    if args.dedupe:
        key = ["source_iso3", "target_iso3", "sentence_text"]
        given = given.drop_duplicates(key)
        received = received.drop_duplicates(key)

    report += [
        f"03_build_panel  ({time.strftime('%Y-%m-%d %H:%M:%S')})   dedupe={args.dedupe}",
        f"mentions in input: {n0}",
        f"  dropped: unparseable date {n_nat}, outside {SAMPLE_START}-{SAMPLE_END} {n_outside}, unscored {n_unscored}",
        f"  kept: given {len(given)}, received {len(received)}",
        "",
    ]

    # ---- full country x quarter grid ----------------------------------- #
    quarters = pd.period_range(q_start, q_end, freq="Q")
    grid = pd.MultiIndex.from_product([SOURCE_COUNTRIES, quarters],
                                      names=["country", "quarter"]).to_frame(index=False)

    g_wide = pivot_side(given, "source_iso3", "target_iso3", "score_", "n_", "n_docs_", quarters)
    r_wide = pivot_side(received, "target_iso3", "source_iso3", "recv_", "n_recv_", "n_docs_recv_", quarters)

    panel = grid.merge(g_wide, on=["country", "quarter"], how="left") \
                .merge(r_wide, on=["country", "quarter"], how="left")

    for c in panel.columns:  # counts: missing cell = 0 mentions
        if c.startswith(("n_", "n_recv_", "n_docs_")):
            panel[c] = panel[c].fillna(0).astype(int)

    # ---- derived measures ---------------------------------------------- #
    panel["tilt"] = panel["score_RUS"] - panel["score_UKR"]
    panel["asymmetry"] = panel["tilt"].abs()
    panel["valence"] = panel[["score_RUS", "score_UKR"]].mean(axis=1, skipna=False)
    panel["balanced"] = np.where(panel["asymmetry"].notna(),
                                 (panel["asymmetry"] < BALANCE_CUTOFF).astype(float), np.nan)
    panel["low_mention"] = (panel[["n_RUS", "n_UKR"]].min(axis=1) < MIN_MENTIONS).astype(int)

    panel["recv_mean"] = panel[["recv_RUS", "recv_UKR"]].mean(axis=1, skipna=False)
    panel["recv_min"] = panel[["recv_RUS", "recv_UKR"]].min(axis=1, skipna=False)
    both = panel["recv_RUS"].notna() & panel["recv_UKR"].notna()
    panel["recv_both_pos"] = np.where(
        both, ((panel["recv_RUS"] > POSITIVE_THRESHOLD) & (panel["recv_UKR"] > POSITIVE_THRESHOLD)).astype(float),
        np.nan)
    panel["low_mention_recv"] = (panel[["n_recv_RUS", "n_recv_UKR"]].min(axis=1) < MIN_MENTIONS).astype(int)

    # ---- metadata ------------------------------------------------------- #
    inv = pd.Period(INVASION_QUARTER, "Q")
    panel["group"] = panel["country"].map(GROUP_OF)
    panel["year"] = panel["quarter"].dt.year
    panel["t"] = (panel["quarter"] - q_start).apply(lambda x: x.n)
    panel["post"] = (panel["quarter"] >= inv).astype(int)
    panel["quarter"] = panel["quarter"].astype(str)  # e.g. "2022Q1" (parquet/csv friendly)

    lead = ["country", "group", "quarter", "year", "t", "post"]
    panel = panel[lead + [c for c in panel.columns if c not in lead]]
    panel = panel.sort_values(["country", "t"]).reset_index(drop=True)

    panel.to_parquet(OUTPUT_DIR / f"03_panel{suffix}.parquet", index=False)
    panel.to_csv(OUTPUT_DIR / f"03_panel{suffix}.csv", index=False)

    # ---- summary -------------------------------------------------------- #
    n_cells = len(panel)
    report += [
        f"country-quarters: {n_cells} ({len(SOURCE_COUNTRIES)} countries x {len(quarters)} quarters)",
        f"  with tilt (both belligerents mentioned): {panel['tilt'].notna().sum()}",
        f"  with both received scores:               {both.sum()}",
        f"  low_mention (<{MIN_MENTIONS} toward a belligerent, given): {int(panel['low_mention'].sum())}",
        "",
        "Coverage by country (quarters with tilt / with joint reception, of "
        f"{len(quarters)}; post-invasion in brackets):",
    ]
    cov = panel.groupby("country").apply(lambda d: pd.Series({
        "group": d["group"].iloc[0],
        "tilt": int(d["tilt"].notna().sum()),
        "tilt_post": int(d.loc[d["post"] == 1, "tilt"].notna().sum()),
        "recv": int(d["recv_mean"].notna().sum()),
        "recv_post": int(d.loc[d["post"] == 1, "recv_mean"].notna().sum()),
        "mentions_given": int(d[["n_RUS", "n_UKR"]].sum().sum()),
        "mentions_recv": int(d[["n_recv_RUS", "n_recv_UKR"]].sum().sum()),
    }), include_groups=False) if "include_groups" in pd.DataFrame.groupby.__code__.co_varnames else \
        panel.groupby("country").apply(lambda d: pd.Series({
            "group": d["group"].iloc[0],
            "tilt": int(d["tilt"].notna().sum()),
            "tilt_post": int(d.loc[d["post"] == 1, "tilt"].notna().sum()),
            "recv": int(d["recv_mean"].notna().sum()),
            "recv_post": int(d.loc[d["post"] == 1, "recv_mean"].notna().sum()),
            "mentions_given": int(d[["n_RUS", "n_UKR"]].sum().sum()),
            "mentions_recv": int(d[["n_recv_RUS", "n_recv_UKR"]].sum().sum()),
        }))
    report += [cov.to_string(), "", "Mentions per year (given / received):"]
    yr = pd.concat({
        "given": given.groupby(given["quarter"].dt.year).size(),
        "received": received.groupby(received["quarter"].dt.year).size(),
    }, axis=1).fillna(0).astype(int)
    report += [yr.to_string(), "", f"runtime: {time.time() - t0:.0f}s"]

    text = "\n".join(report)
    (OUTPUT_DIR / f"03_summary{suffix}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()