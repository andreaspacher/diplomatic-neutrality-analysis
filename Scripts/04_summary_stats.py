#!/usr/bin/env python3
"""
04_summary_stats.py — descriptive tables from the country-quarter panel.

Tables (post-invasion = quarters >= INVASION_QUARTER, country-quarter means):
  1 groups      measures by neutrality group                 (cf. paper tab:groups)
  2 countries   measures by country                          (cf. paper tab:countries)
  3 levels      mean sentiment toward / received from RUS & UKR by country
  4 prepost     asymmetry / tilt / valence by group, pre vs post
  5 coverage    quarters with data and mention counts per country (thin baselines!)

Outputs (Output/): 04_table_<name>.csv (numeric), 04_table_<name>.tex (booktabs),
04_summary.txt, 04_numbers.tex (macros for in-text numbers, e.g. \\statAsymDipl).

Usage: python Scripts/04_summary_stats.py [--dedupe]
"""

from __future__ import annotations

import argparse

import pandas as pd

from config import GROUP_OF, GROUPS, OUTPUT_DIR
from utils import (COUNTRY_NAMES, COUNTRY_ORDER, GROUP_CAP, GROUP_ORDER, GROUP_SHORT, fmt,
                   latex_table, load_panel, num, pct, write_macros)


def save(name, df, tex, sfx):
    df.to_csv(OUTPUT_DIR / f"04_table_{name}{sfx}.csv")
    (OUTPUT_DIR / f"04_table_{name}{sfx}.tex").write_text(tex, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()
    sfx = "_dedup" if args.dedupe else ""

    panel = load_panel(args.dedupe)
    post = panel[panel["post"] == 1]
    txt, macros = [], {}

    # ------------------------------------------------------ 1 groups ------
    g = post.groupby("group")
    t1 = pd.DataFrame({
        "asymmetry": g["asymmetry"].mean(), "balanced": g["balanced"].mean(),
        "valence": g["valence"].mean(), "recv_mean": g["recv_mean"].mean(),
        "recv_min": g["recv_min"].mean(), "both_pos": g["recv_both_pos"].mean(),
        "n_tilt": g["tilt"].count(), "n_recv": g["recv_mean"].count(),
    }).reindex(GROUP_ORDER)

    spec = [("Mean asymmetry (absolute tilt)", "asymmetry", fmt),
            ("Share of quarters balanced (asymmetry below 0.25)", "balanced", pct),
            ("Mean valence toward both belligerents", "valence", fmt),
            ("Mean received sentiment (from Russia and Ukraine)", "recv_mean", fmt),
            ("Minimum received sentiment (from Russia and Ukraine)", "recv_min", fmt),
            ("Both belligerents positive", "both_pos", pct)]
    rows = [[lab] + [f(t1.loc[grp, col]) for grp in GROUP_ORDER] for lab, col, f in spec]
    rows.append(["Country-quarters (tilt / received)"] +
                [f"{int(t1.loc[grp, 'n_tilt'])} / {int(t1.loc[grp, 'n_recv'])}" for grp in GROUP_ORDER])
    save("groups", t1, latex_table(["Measure"] + [GROUP_SHORT[x] for x in GROUP_ORDER], rows,
                                   "Post-invasion quarters, by group (country-quarter means).",
                                   "tab:groups", "lccc"), sfx)
    txt += ["== Table 1: groups (post-invasion) ==", t1.round(3).to_string(), ""]
    for grp in GROUP_ORDER:
        for col, M, is_pct in [("asymmetry", "Asym", 0), ("balanced", "Bal", 1), ("valence", "Val", 0),
                               ("recv_mean", "RecvMean", 0), ("recv_min", "RecvMin", 0), ("both_pos", "BothPos", 1)]:
            v = t1.loc[grp, col]
            macros[f"stat{M}{GROUP_CAP[grp]}"] = f"{100 * v:.0f}" if is_pct else num(v)

    # ---------------------------------------------------- 2 countries -----
    recs, rows = [], []
    last_g = None
    for c in COUNTRY_ORDER:
        d = post[post["country"] == c]
        r = dict(country=c, group=GROUP_OF[c], asymmetry=d["asymmetry"].mean(), tilt=d["tilt"].mean(),
                 valence=d["valence"].mean(), n_balanced=int(d["balanced"].sum()), n_tilt=int(d["tilt"].count()),
                 recv_min=d["recv_min"].mean(), both_pos=d["recv_both_pos"].mean(),
                 n_recv=int(d["recv_mean"].count()))
        recs.append(r)
        if last_g is not None and r["group"] != last_g:
            rows.append("midrule")
        rows.append([GROUP_SHORT[r["group"]] if r["group"] != last_g else "", COUNTRY_NAMES[c],
                     fmt(r["asymmetry"]), fmt(r["tilt"], plus=True), fmt(r["valence"]),
                     f"{r['n_balanced']}/{r['n_tilt']}", fmt(r["recv_min"]), pct(r["both_pos"])])
        last_g = r["group"]
    t2 = pd.DataFrame(recs).set_index("country")
    save("countries", t2, latex_table(
        ["Neutrality", "Country", "Asymmetry", "Tilt\\textsuperscript{a}", "Valence", "Balanced",
         "Min.\\ received", "Both positive"], rows, "Post-invasion quarters, by country.",
        "tab:countries", "llcccccc",
        note="\\textsuperscript{a} Positive values indicate pro-Russian lean, negative values a pro-Ukrainian lean. "
             "Balanced = quarters with asymmetry below 0.25 / quarters with tilt."), sfx)
    txt += ["== Table 2: countries (post-invasion) ==", t2.round(3).to_string(), ""]

    # ------------------------------------------------------ 3 levels ------
    t3 = (post.groupby("country")[["score_RUS", "score_UKR", "recv_RUS", "recv_UKR"]].mean()
              .reindex(COUNTRY_ORDER))
    rows = [[COUNTRY_NAMES[c]] + [fmt(t3.loc[c, k], plus=True) for k in t3.columns] for c in COUNTRY_ORDER]
    save("levels", t3, latex_table(
        ["Country", "Toward Russia", "Toward Ukraine", "From Russia", "From Ukraine"], rows,
        "Mean sentiment toward and received from the belligerents, post-invasion quarters.",
        "tab:levels", "lcccc"), sfx)
    txt += ["== Table 3: sentiment levels (post-invasion) ==", t3.round(3).to_string(), ""]

    # ------------------------------------------------------ 4 pre/post ----
    rec = []
    for grp in GROUP_ORDER:
        for per, d in [("pre", panel[(panel["post"] == 0) & (panel["group"] == grp)]),
                       ("post", panel[(panel["post"] == 1) & (panel["group"] == grp)])]:
            rec.append(dict(group=grp, period=per, asymmetry=d["asymmetry"].mean(), tilt=d["tilt"].mean(),
                            valence=d["valence"].mean(), n_tilt=int(d["tilt"].count())))
    t4 = pd.DataFrame(rec).set_index(["group", "period"])
    rows = [[GROUP_SHORT[grp], per, fmt(t4.loc[(grp, per), "asymmetry"]), fmt(t4.loc[(grp, per), "tilt"], plus=True),
             fmt(t4.loc[(grp, per), "valence"]), str(int(t4.loc[(grp, per), "n_tilt"]))]
            for grp in GROUP_ORDER for per in ("pre", "post")]
    save("prepost", t4, latex_table(["Group", "Period", "Asymmetry", "Tilt", "Valence", "$n$"], rows,
                                    "Sentiment toward the belligerents before and after the invasion.",
                                    "tab:prepost", "llcccc"), sfx)
    txt += ["== Table 4: pre vs post ==", t4.round(3).to_string(), ""]

    # ------------------------------------------------------ 5 coverage ----
    rec = []
    for c in COUNTRY_ORDER:
        d = panel[panel["country"] == c]
        rec.append(dict(
            country=c, group=GROUP_OF[c],
            tilt_pre=int(d.loc[d["post"] == 0, "tilt"].notna().sum()),
            tilt_post=int(d.loc[d["post"] == 1, "tilt"].notna().sum()),
            recv_pre=int(d.loc[d["post"] == 0, "recv_mean"].notna().sum()),
            recv_post=int(d.loc[d["post"] == 1, "recv_mean"].notna().sum()),
            low_mention_quarters=int(d["low_mention"].sum()),
            mentions_given=int(d[["n_RUS", "n_UKR"]].sum().sum()),
            mentions_received=int(d[["n_recv_RUS", "n_recv_UKR"]].sum().sum())))
    t5 = pd.DataFrame(rec).set_index("country")
    rows = [[COUNTRY_NAMES[c], str(t5.loc[c, "tilt_pre"]), str(t5.loc[c, "tilt_post"]),
             str(t5.loc[c, "recv_pre"]), str(t5.loc[c, "recv_post"]),
             f"{t5.loc[c, 'mentions_given']:,}".replace(",", "\\,"),
             f"{t5.loc[c, 'mentions_received']:,}".replace(",", "\\,")] for c in COUNTRY_ORDER]
    save("coverage", t5, latex_table(
        ["Country", "Tilt pre", "Tilt post", "Recv.\\ pre", "Recv.\\ post", "Mentions given", "Mentions received"],
        rows, "Data coverage: quarters with data and mention counts.", "tab:coverage", "lcccccc"), sfx)
    txt += ["== Table 5: coverage ==", t5.to_string(), ""]

    # ------------------------------------------------------ totals --------
    n_cells = len(panel)
    macros.update({
        "statCellsTotal": str(n_cells),
        "statCellsTilt": str(int(panel["tilt"].notna().sum())),
        "statCellsRecv": str(int(panel["recv_mean"].notna().sum())),
        "statMentionsGiven": f"{int(panel[['n_RUS', 'n_UKR']].sum().sum()):,}".replace(",", "\\,"),
        "statMentionsRecv": f"{int(panel[['n_recv_RUS', 'n_recv_UKR']].sum().sum()):,}".replace(",", "\\,"),
    })
    txt += [f"country-quarters: {n_cells}, with tilt: {macros['statCellsTilt']}, "
            f"with joint reception: {macros['statCellsRecv']}"]

    write_macros(macros, OUTPUT_DIR / f"04_numbers{sfx}.tex")
    text = "\n".join(txt)
    (OUTPUT_DIR / f"04_summary{sfx}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()