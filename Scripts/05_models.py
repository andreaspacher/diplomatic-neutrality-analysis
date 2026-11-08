#!/usr/bin/env python3
"""
05_models.py — estimates H1-H4 on the country-quarter panel (functions live in
hypotheses.py so that 06_robustness.py can reuse them).

  H1 reality        asymmetry / balanced / valence ~ group + quarter FE   (post quarters)
  H2 invariance     DiD around the invasion on asymmetry and tilt; TOST equivalence
  H3 reciprocity    received sentiment ~ tilt x group + country FE + quarter FE
  H4 joint recept.  received mean / min / both-positive ~ group + quarter FE (post quarters)

All: OLS, HC1 standard errors. Group contrasts = comparison group MINUS diplomatic.

Outputs (Output/): 05_results.csv (long format), 05_results.txt,
05_tab_recip.tex (cf. paper tab:recip), 05_numbers.tex (macros, e.g. \\hOneAsymMil, ...Est/Se/P)

Usage: python Scripts/05_models.py [--dedupe] [--h3-sample all|post]
                                   [--margin-asym 0.15] [--margin-tilt 0.20]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from hypotheses import h1, h2, h3, h4
from utils import fmt, latex_table, load_panel, num, pfmt, write_macros


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--h3-sample", choices=["all", "post"], default="all")
    ap.add_argument("--margin-asym", type=float, default=0.15, help="TOST equivalence margin, asymmetry")
    ap.add_argument("--margin-tilt", type=float, default=0.20, help="TOST equivalence margin, tilt")
    args = ap.parse_args()
    sfx = "_dedup" if args.dedupe else ""

    panel = load_panel(args.dedupe)
    rows = (h1(panel) + h2(panel, args.margin_asym, args.margin_tilt)
            + h3(panel, args.h3_sample) + h4(panel))
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / f"05_results{sfx}.csv", index=False)

    # ---- readable report ----------------------------------------------- #
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    txt = [f"05_models  dedupe={args.dedupe}  h3_sample={args.h3_sample}  "
           f"margins: asym {args.margin_asym}, tilt {args.margin_tilt}",
           "contrasts = comparison group minus diplomatic; OLS + HC1; p-values approximate (few clusters)"]
    for hyp, d in df.groupby("hyp"):
        txt += ["", f"=== {hyp} ===",
                d[["outcome", "term", "est", "se", "p", "lo", "hi", "n", "note"]]
                .round(4).to_string(index=False)]
    text = "\n".join(txt)
    (OUTPUT_DIR / f"05_results{sfx}.txt").write_text(text, encoding="utf-8")
    print(text)

    # ---- tab:recip ------------------------------------------------------ #
    sl = df[(df["hyp"] == "H3") & df["term"].str.startswith("slope:")].copy()
    sl["group"] = sl["term"].str.split(":").str[1]
    cols = ["diplomatic", "military", "former"]
    rows_t = []
    for y, lab in [("recv_UKR", "Ukraine"), ("recv_RUS", "Russia")]:
        cells = []
        for g in cols:
            r = sl[(sl["outcome"] == y) & (sl["group"] == g)]
            cells.append(f"{fmt(r['est'].iloc[0])} ({r['se'].iloc[0]:.2f})" if len(r) else "--")
        rows_t.append([lab] + cells)
    tex = latex_table(["Received from", "Diplomatic", "Military", "Former"], rows_t,
                      "Slope of received sentiment on tilt, by group (HC1 standard errors in parentheses).",
                      "tab:recip", "lccc")
    (OUTPUT_DIR / f"05_tab_recip{sfx}.tex").write_text(tex, encoding="utf-8")

    # ---- macros ---------------------------------------------------------- #
    macros = {}
    for r in df[df["key"].notna()].itertuples():
        if not np.isnan(r.est):
            macros[f"{r.key}Est"] = num(r.est)
            macros[f"{r.key}Se"] = f"{r.se:.2f}"
        macros[f"{r.key}P"] = pfmt(r.p)
    write_macros(macros, OUTPUT_DIR / f"05_numbers{sfx}.tex")
    print(f"\nWrote {len(macros)} macros to 05_numbers{sfx}.tex")


if __name__ == "__main__":
    main()