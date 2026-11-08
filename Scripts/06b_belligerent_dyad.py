#!/usr/bin/env python3
"""
06b_belligerent_dyad.py — benchmark: how do Russia and Ukraine speak about EACH OTHER,
compared with how they speak about the neutrals?

Needs the "dyadic" mentions (01 extended + 02 rerun). Uses the same filters as 03.

Outputs (Output/): 06b_dyad_table.csv, 06b_dyad_summary.txt
Figure: figures/figA2_dyad_benchmark.<fmt>   quarterly mean sentiment, mention-level

Usage: python Scripts/06b_belligerent_dyad.py [--dedupe] [--formats pdf png]
"""

from __future__ import annotations

import argparse
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from config import (FIGURES_DIR, GROUP_OF, INVASION_QUARTER, OUTPUT_DIR, SAMPLE_END,
                    SAMPLE_START, SOURCE_COUNTRIES)
from scipy import stats

MENTIONS = OUTPUT_DIR / "02_mentions_absa.parquet"
BEL = ["RUS", "UKR"]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def load(dedupe: bool) -> pd.DataFrame:
    parse_quarters = importlib.import_module("03_build_panel").parse_quarters
    m = pd.read_parquet(MENTIONS, columns=["direction", "source_iso3", "target_iso3", "date",
                                           "sentence_text", "absa_score", "absa_label"])
    m["quarter"] = parse_quarters(m["date"])
    q0, q1 = pd.Period(SAMPLE_START, "Q"), pd.Period(SAMPLE_END, "Q")
    m = m[m["quarter"].notna() & m["absa_score"].notna() & (m["quarter"] >= q0) & (m["quarter"] <= q1)]
    dy = ((m["direction"] == "dyadic") & m["source_iso3"].isin(BEL) & m["target_iso3"].isin(BEL)
          & (m["source_iso3"] != m["target_iso3"]))
    rc = (m["direction"] == "received") & m["source_iso3"].isin(BEL) & m["target_iso3"].isin(SOURCE_COUNTRIES)
    m = m[dy | rc].copy()
    if dedupe:
        m = m.drop_duplicates(["direction", "source_iso3", "target_iso3", "sentence_text"])
    m["post"] = m["quarter"] >= pd.Period(INVASION_QUARTER, "Q")
    m["audience"] = np.where(m["direction"] == "dyadic", "belligerent",
                             m["target_iso3"].map(GROUP_OF).fillna("?"))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--formats", nargs="+", default=["pdf"])
    args = ap.parse_args()
    sfx = "_dedup" if args.dedupe else ""

    m = load(args.dedupe)
    if not (m["direction"] == "dyadic").any():
        raise SystemExit("No dyadic mentions found. Extend 01 (RUS<->UKR), rerun 01 and 02 first.")

    tab = (m.groupby(["source_iso3", "audience", "post"])
             .agg(n=("absa_score", "size"), mean=("absa_score", "mean"),
                  share_neg=("absa_label", lambda s: (s == "Negative").mean()),
                  share_pos=("absa_label", lambda s: (s == "Positive").mean())))
    tab.to_csv(OUTPUT_DIR / f"06b_dyad_table{sfx}.csv")
    text = "\n".join(["06b_belligerent_dyad  (speaker x audience x post-invasion)",
                      "audience: belligerent = the other belligerent; diplomatic / military / former = neutral groups",
                      "", tab.round(3).to_string()])
    (OUTPUT_DIR / f"06b_dyad_summary{sfx}.txt").write_text(text, encoding="utf-8")
    print(text)

    # ---- figure: quarterly mention-level means ---------------------------- #
    m["line"] = m["source_iso3"] + " about " + np.where(m["audience"] == "belligerent",
                                                            m["target_iso3"], "neutrals")
    q = m.groupby(["line", "quarter"])["absa_score"].mean().unstack("line")
    q.index = q.index.start_time
    # ---- numbers for the text -------------------------------------------- #
    post = m[m["post"]].copy()
    post["kind"] = np.where(post["audience"] == "belligerent", "belligerent", "neutrals")
    pooled = (post.groupby(["source_iso3", "kind"])
                  .agg(n=("absa_score", "size"), mean=("absa_score", "mean"),
                       share_neg=("absa_label", lambda s: (s == "Negative").mean())))
    lines = ["", "Pooled post-invasion means (mention level; neutrals = all three groups combined):",
             pooled.round(3).to_string()]
    for spk in BEL:
        other = "UKR" if spk == "RUS" else "RUS"
        a, b = f"{spk} about {other}", f"{spk} about neutrals"
        gap = pooled.loc[(spk, "neutrals"), "mean"] - pooled.loc[(spk, "belligerent"), "mean"]
        d = (q[a] - q[b]).dropna()                       # quarterly difference: dyad minus neutrals
        k = int((d < 0).sum())
        p_sign = stats.binomtest(k, len(d), 0.5).pvalue
        p_wil = stats.wilcoxon(d).pvalue
        lines.append(f"{spk}: gap (neutrals - dyad, post) = {gap:.2f}; dyad more negative than neutrals in "
                     f"{k} of {len(d)} quarters (sign test p = {p_sign:.2g}; Wilcoxon p = {p_wil:.2g})")
    extra = "\n".join(lines)
    print(extra)
    with open(OUTPUT_DIR / f"06b_dyad_summary{sfx}.txt", "a", encoding="utf-8") as fh:
        fh.write("\n" + extra)

    style = {"RUS about UKR": ("#b2182b", "-"), "UKR about RUS": ("#2166ac", "-"),
             "RUS about neutrals": ("#b2182b", ":"), "UKR about neutrals": ("#2166ac", ":")}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    handles = {}
    for col, (c, ls) in style.items():
        if col in q:
            handles[col], = ax.plot(q.index, q[col], color=c, ls=ls, marker="o", ms=3, lw=1.2)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.axvline(pd.Timestamp("2022-02-24"), color="k", ls="--", lw=0.8)
    ax.set_ylabel("Mean sentiment toward the target")
    ax.spines[["top", "right"]].set_visible(False)

    # legend to the right: dotted lines first, a blank row, then the solid lines
    order = ["RUS about neutrals", "UKR about neutrals", None, "RUS about UKR", "UKR about RUS"]
    leg_handles, leg_labels = [], []
    for key in order:
        if key is None:
            leg_handles.append(Line2D([], [], color="none"))   # invisible spacer entry
            leg_labels.append(" ")
        elif key in handles:
            leg_handles.append(handles[key])
            leg_labels.append(key)
    ax.legend(leg_handles, leg_labels, frameon=False, fontsize=7.5, loc="center left",
              bbox_to_anchor=(1.02, 0.5), borderaxespad=0, handlelength=2.5)
    fig.tight_layout()

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for f in args.formats:
        path = FIGURES_DIR / f"figA2_dyad_benchmark{sfx}.{f}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()