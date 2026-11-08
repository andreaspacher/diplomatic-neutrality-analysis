#!/usr/bin/env python3
"""
04b_data_overview.py — how much data is there? (opening of the Results section)

Tables (Output/):
  04b_table_mentions.csv/.tex  per country x direction: mentions pre / post / total, distinct documents
  04b_table_cells.csv/.tex     distribution of mentions per country-quarter-belligerent cell
Figure:
  figures/figA1_mention_coverage.<fmt>  heatmap country x quarter of mentions toward the
                                        LESS-mentioned belligerent (given) / from the
                                        less-mentioning belligerent (received)
Also: 04b_overview.txt, 04b_numbers.tex (macros, e.g. \\ovMentionsTotal)

Mention-level counts use Output/02_mentions_absa.parquet with the same filters as
03_build_panel.py (valid date in sample window, scored, valid direction/country pair).

Usage: python Scripts/04b_data_overview.py [--formats pdf png]
"""

from __future__ import annotations

import argparse
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from config import (FIGURES_DIR, GROUP_OF, INVASION_QUARTER, MIN_MENTIONS, OUTPUT_DIR,
                    RECEIVING_COUNTRIES, SAMPLE_END, SAMPLE_START, SOURCE_COUNTRIES)
from utils import COUNTRY_NAMES, COUNTRY_ORDER, latex_table, load_panel, write_macros

MENTIONS = OUTPUT_DIR / "02_mentions_absa.parquet"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def ti(x) -> str:  # thousands separator for LaTeX
    return f"{int(x):,}".replace(",", "\\,")


def load_mentions(dedupe: bool = False) -> pd.DataFrame:
    parse_quarters = importlib.import_module("03_build_panel").parse_quarters
    m = pd.read_parquet(MENTIONS, columns=["file", "doc_id", "direction", "source_iso3", "target_iso3",
                                           "date", "sentence_idx", "sentence_text", "absa_score"])
    m["quarter"] = parse_quarters(m["date"])
    q0, q1 = pd.Period(SAMPLE_START, "Q"), pd.Period(SAMPLE_END, "Q")
    m = m[m["quarter"].notna() & m["absa_score"].notna() & (m["quarter"] >= q0) & (m["quarter"] <= q1)]
    g = ((m["direction"] == "given") & m["source_iso3"].isin(SOURCE_COUNTRIES)
         & m["target_iso3"].isin(RECEIVING_COUNTRIES))
    r = ((m["direction"] == "received") & m["source_iso3"].isin(RECEIVING_COUNTRIES)
         & m["target_iso3"].isin(SOURCE_COUNTRIES))
    m = m[g | r].copy()
    if dedupe:
        m = m.drop_duplicates(["direction", "source_iso3", "target_iso3", "sentence_text"])
    m["country"] = np.where(m["direction"] == "given", m["source_iso3"], m["target_iso3"])
    m["period"] = np.where(m["quarter"] >= pd.Period(INVASION_QUARTER, "Q"), "post", "pre")
    m["doc_key"] = m["file"].astype(str) + "|" + m["doc_id"].astype(str)
    m["sent_key"] = m["doc_key"] + "#" + m["sentence_idx"].astype(str)
    return m


def table_mentions(m):
    base = m.groupby(["country", "direction"]).agg(total=("doc_key", "size"), docs=("doc_key", "nunique"),
                                                   sentences=("sent_key", "nunique"))
    pp = m.groupby(["country", "direction", "period"]).size().unstack("period").reindex(columns=["pre", "post"])
    A = base.join(pp).fillna(0).astype(int)
    A = A.reindex(pd.MultiIndex.from_product([COUNTRY_ORDER, ["given", "received"]],
                                             names=["country", "direction"]), fill_value=0)
    A = A[["pre", "post", "total", "docs", "sentences"]]

    rows = []
    for c in COUNTRY_ORDER:
        rows.append([COUNTRY_NAMES[c]] + [ti(A.loc[(c, d), k]) for d in ("given", "received")
                                          for k in ("pre", "post", "total", "docs")])
    tot = {d: A.xs(d, level="direction").sum() for d in ("given", "received")}
    dist_docs = {d: m.loc[m["direction"] == d, "doc_key"].nunique() for d in ("given", "received")}
    rows += ["midrule", ["Total"] + [ti(tot[d][k]) if k != "docs" else ti(dist_docs[d])
                                    for d in ("given", "received") for k in ("pre", "post", "total", "docs")]]
    n_pre = len(pd.period_range(SAMPLE_START, pd.Period(INVASION_QUARTER, "Q") - 1, freq="Q"))
    n_post = len(pd.period_range(INVASION_QUARTER, SAMPLE_END, freq="Q"))
    tex = latex_table(
        ["Country", "Pre", "Post", "Total", "Docs", "Pre", "Post", "Total", "Docs"], rows,
        "Sentence-level mentions by country: given (country $\\rightarrow$ Russia/Ukraine) and "
        "received (Russia/Ukraine $\\rightarrow$ country).", "tab:mentions", "lrrrrrrrr",
        note=f"Pre-invasion: {n_pre} quarters; post-invasion: {n_post} quarters (compare per quarter, not in totals). "
             "Docs = distinct source documents containing at least one mention.")
    # two-level header row is cheap to add by hand: \multicolumn{4}{c}{Given} & \multicolumn{4}{c}{Received}
    return A, tot, dist_docs, tex


def table_cells(panel):
    parts = []
    for d, cols in [("given", {"n_RUS": "RUS", "n_UKR": "UKR"}),
                    ("received", {"n_recv_RUS": "RUS", "n_recv_UKR": "UKR"})]:
        for col, bel in cols.items():
            parts.append(panel[["country", "quarter", "post", col]].rename(columns={col: "n"})
                         .assign(direction=d, belligerent=bel))
    cells = pd.concat(parts)
    cells["period"] = np.where(cells["post"] == 1, "post", "pre")
    out = (cells.groupby(["direction", "period"])["n"]
                .agg(cells="size", mean="mean", median="median", max="max",
                     share_zero=lambda s: (s == 0).mean(),
                     share_below=lambda s: (s < MIN_MENTIONS).mean())
                .reindex([("given", "pre"), ("given", "post"), ("received", "pre"), ("received", "post")]))
    rows = [[d.capitalize(), p, str(int(r["cells"])), f"{r['mean']:.1f}", f"{r['median']:.0f}", ti(r["max"]),
             f"{100 * r['share_zero']:.0f}\\%", f"{100 * r['share_below']:.0f}\\%"]
            for (d, p), r in out.iterrows()]
    tex = latex_table(["Direction", "Period", "Cells", "Mean", "Median", "Max", "Empty", f"$<{MIN_MENTIONS}$"],
                      rows, "Mentions per country-quarter-belligerent cell.", "tab:cells", "llrrrrrr",
                      note=f"A cell is one country, quarter and belligerent. Empty = no mention; "
                           f"$<{MIN_MENTIONS}$ = fewer than {MIN_MENTIONS} mentions (including empty cells).")
    return out, tex


def heatmap(panel, formats, sfx=""):
    qs = sorted(panel["quarter"].unique())
    inv_idx = qs.index(INVASION_QUARTER)
    edges = [-0.5, 0.5, MIN_MENTIONS - 0.5, 19.5, 49.5, 1e9]
    labels = ["0", f"1\u2013{MIN_MENTIONS - 1}", f"{MIN_MENTIONS}\u201319", "20\u201349", "50+"]
    cmap = ListedColormap(["#e6e6e6", "#fde0c5", "#9ecae1", "#4292c6", "#084594"])
    norm = BoundaryNorm(edges, cmap.N)

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True)
    for ax, (title, a, b) in zip(axes, [("Given: mentions toward the less-mentioned belligerent", "n_RUS", "n_UKR"),
                                        ("Received: mentions from the less-mentioning belligerent", "n_recv_RUS", "n_recv_UKR")]):
        mat = (panel.assign(v=panel[[a, b]].min(axis=1)).pivot(index="country", columns="quarter", values="v")
                    .reindex(index=COUNTRY_ORDER, columns=qs))
        ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.axvline(inv_idx - 0.5, color="k", ls="--", lw=0.8)
        for i in range(1, len(COUNTRY_ORDER)):
            if GROUP_OF[COUNTRY_ORDER[i]] != GROUP_OF[COUNTRY_ORDER[i - 1]]:
                ax.axhline(i - 0.5, color="white", lw=2)
        ax.set_yticks(range(len(COUNTRY_ORDER)))
        ax.set_yticklabels([COUNTRY_NAMES[c] for c in COUNTRY_ORDER], fontsize=7.5)
        ax.set_xticks([i for i, q in enumerate(qs) if q.endswith("Q1")])
        ax.set_xticklabels([q[:4] for q in qs if q.endswith("Q1")], fontsize=8)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[1].set_xlabel("Quarter")
    fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(cmap.colors, labels)],
               title="Mentions per quarter", loc="lower center", ncol=5, frameon=False, fontsize=7.5,
               title_fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for f in formats:
        path = FIGURES_DIR / f"figA1_mention_coverage{sfx}.{f}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  wrote {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--formats", nargs="+", default=["pdf"])
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()

    if not MENTIONS.exists():
        raise SystemExit(f"{MENTIONS} not found. Run 02_score_absa.py first.")
    sfx = "_dedup" if args.dedupe else ""
    panel = load_panel(args.dedupe)
    m = load_mentions(args.dedupe)

    A, tot, dist_docs, tex_m = table_mentions(m)
    A.to_csv(OUTPUT_DIR / f"04b_table_mentions{sfx}.csv")
    (OUTPUT_DIR / f"04b_table_mentions{sfx}.tex").write_text(tex_m, encoding="utf-8")

    cells, tex_c = table_cells(panel)
    cells.to_csv(OUTPUT_DIR / f"04b_table_cells{sfx}.csv")
    (OUTPUT_DIR / f"04b_table_cells{sfx}.tex").write_text(tex_c, encoding="utf-8")

    heatmap(panel, args.formats, sfx)

    # sanity check: panel counts must match mention-level counts
    n_panel = int(panel[["n_RUS", "n_UKR", "n_recv_RUS", "n_recv_UKR"]].sum().sum())
    check = "OK" if n_panel == len(m) else f"MISMATCH (panel {n_panel} vs mentions {len(m)}; fine if the panel was built with --dedupe)"

    post_q = len(pd.period_range(INVASION_QUARTER, SAMPLE_END, freq="Q"))
    pre_q = len(pd.period_range(SAMPLE_START, pd.Period(INVASION_QUARTER, "Q") - 1, freq="Q"))
    macros = {
        "ovMentionsTotal": ti(len(m)),
        "ovMentionsGiven": ti(tot["given"]["total"]),
        "ovMentionsRecv": ti(tot["received"]["total"]),
        "ovMentionsPre": ti(tot["given"]["pre"] + tot["received"]["pre"]),
        "ovMentionsPost": ti(tot["given"]["post"] + tot["received"]["post"]),
        "ovDocsTotal": ti(m["doc_key"].nunique()),
        "ovSentencesTotal": ti(m["sent_key"].nunique()),
        "ovPerQuarterPre": f"{(tot['given']['pre'] + tot['received']['pre']) / pre_q:.0f}",
        "ovPerQuarterPost": f"{(tot['given']['post'] + tot['received']['post']) / post_q:.0f}",
        "ovMedianCellGiven": f"{cells.loc[('given', 'post'), 'median']:.0f}",
        "ovMedianCellRecv": f"{cells.loc[('received', 'post'), 'median']:.0f}",
        "ovShareThinGiven": f"{100 * cells.loc[('given', 'post'), 'share_below']:.0f}",
        "ovShareThinRecv": f"{100 * cells.loc[('received', 'post'), 'share_below']:.0f}",
    }
    write_macros(macros, OUTPUT_DIR / f"04b_numbers{sfx}.tex")

    txt = ["04b_data_overview", f"mention-level vs panel counts: {check}", "",
           "Mentions per country and direction:", A.to_string(), "",
           f"distinct documents: given {dist_docs['given']}, received {dist_docs['received']}, "
           f"overall {m['doc_key'].nunique()}", "",
           "Distribution per country-quarter-belligerent cell:", cells.round(3).to_string()]
    text = "\n".join(txt)
    (OUTPUT_DIR / f"04b_overview{sfx}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()