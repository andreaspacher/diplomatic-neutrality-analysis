#!/usr/bin/env python3
"""
07_figures.py — the paper's figures, built from Output/03_panel.parquet.

  fig1  equidistance scatter (post-invasion): sentiment toward Ukraine (x) vs Russia (y)
  fig2  asymmetry (absolute tilt) over time, group means
  fig3  joint reception quadrant: mean sentiment received from Russia (x) / Ukraine (y),
        post-invasion, country points + group "clouds" + large group-mean symbols
  fig4  tilt over time by group, centred on each country's baseline mean
        (anticipation window shaded)

Output: figures/<name>.<fmt>  (names are set in FIG_NAMES below so they match the paper's
\\includegraphics paths).

Usage:
  python Scripts/07_figures.py
  python Scripts/07_figures.py --figs 3 4 --formats pdf png
  python Scripts/07_figures.py --no-demean     # fig4 on raw tilt
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull

from config import BALANCE_CUTOFF, FIGURES_DIR, GROUP_LABELS, GROUP_OF, INVASION_QUARTER
from utils import COUNTRY_NAMES, GROUP_ORDER, load_panel

# ------------------------------------------------------------------ settings
FIG_NAMES = {1: "fig1_equidistance", 2: "fig2_asymmetry_time",
             3: "fig3_reception", 4: "fig4_tilt_by_group"}   # edit to match the paper's file names
ANTICIPATION_START = "2021Q3"                                # shaded window in fig4 (to the invasion)
WAR_DATE = pd.Timestamp("2022-02-24")

GROUP_COLOR = {"diplomatic": "#0072B2", "military": "#D55E00", "former": "#555555"}
GROUP_MARKER = {"diplomatic": "o", "military": "^", "former": "s"}
GROUP_LINESTYLE = {"diplomatic": "-", "military": "--", "former": ":"}
GROUP_LABEL = GROUP_LABELS

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "savefig.dpi": 300})


# ------------------------------------------------------------------- helpers
def prep(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    d["quarter_ts"] = pd.PeriodIndex(d["quarter"], freq="Q").start_time
    return d


def present_groups(d: pd.DataFrame) -> list[str]:
    return [g for g in GROUP_ORDER if g in set(d["group"])]


def save(fig, key: int, formats):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for f in formats:
        path = FIGURES_DIR / f"{FIG_NAMES[key]}.{f}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


def group_handles(groups):
    return [Line2D([], [], marker=GROUP_MARKER[g], color="w", markerfacecolor=GROUP_COLOR[g],
                   markeredgecolor="black", markeredgewidth=0.4, markersize=7, label=GROUP_LABEL[g])
            for g in groups]


def place_labels(ax, items, fontsize=8, color="0.2"):
    """items: list of (x, y, text). Uses adjustText if installed, else small fixed offsets."""
    texts = [ax.text(x, y, t, fontsize=fontsize, color=color, zorder=6) for x, y, t in items]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666"))
    except Exception:
        xr, yr = np.diff(ax.get_xlim())[0], np.diff(ax.get_ylim())[0]
        for t in texts:
            x, y = t.get_position()
            t.set_position((x + 0.012 * xr, y + 0.012 * yr))
    return texts


def cloud(ax, pts, color, rx, ry, ls="-", n=24):
    """Rounded outline (padded convex hull) around the points; extent of a group."""
    pts = np.asarray(pts, float)
    if len(pts) == 0:
        return
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.concatenate([np.c_[p[0] + rx * np.cos(th), p[1] + ry * np.sin(th)] for p in pts])
    poly = ring[ConvexHull(ring).vertices]
    ax.fill(poly[:, 0], poly[:, 1], facecolor=color, alpha=0.10, edgecolor="none", zorder=1)
    closed = np.vstack([poly, poly[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, ls=ls, lw=0.9, alpha=0.8, zorder=2)


def spread_labels(vals, gap):
    """Nudge label y-positions apart so that none are closer than `gap`."""
    vals = np.asarray(vals, float)
    order = np.argsort(vals)
    adj = vals[order].copy()
    for i in range(1, len(adj)):
        if adj[i] - adj[i - 1] < gap:
            adj[i] = adj[i - 1] + gap
    adj += vals[order].mean() - adj.mean()
    out = np.empty_like(adj)
    out[order] = adj
    return out


def time_axis(ax):
    years = pd.date_range("2020-01-01", "2025-01-01", freq="YS")
    ax.set_xticks(years)
    ax.set_xticklabels([str(y.year) for y in years], fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)


# ------------------------------------------------------- fig 1: equidistance
def fig1(d, formats, **_):
    p = d[d["post"] == 1].dropna(subset=["score_RUS", "score_UKR"])
    groups = present_groups(p)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    xs = np.array([-1.05, 1.05])
    ax.fill_between(xs, xs - BALANCE_CUTOFF, xs + BALANCE_CUTOFF, color="#dddddd", alpha=0.7, lw=0, zorder=0)
    ax.plot(xs, xs, color="k", lw=0.8, ls="--", zorder=1)
    ax.axhline(0, color="#999999", lw=0.5, zorder=1)
    ax.axvline(0, color="#999999", lw=0.5, zorder=1)
    for g in groups:
        s = p[p["group"] == g]
        ax.scatter(s["score_UKR"], s["score_RUS"], s=14, c=GROUP_COLOR[g], marker=GROUP_MARKER[g],
                   alpha=0.30, lw=0, zorder=2)
    mean = p.groupby("country")[["score_UKR", "score_RUS"]].mean()
    items = []
    for c, r in mean.iterrows():
        g = GROUP_OF[c]
        ax.scatter(r["score_UKR"], r["score_RUS"], s=70, c=GROUP_COLOR[g], marker=GROUP_MARKER[g],
                   edgecolor="white", lw=0.8, zorder=4)
        items.append((r["score_UKR"], r["score_RUS"], COUNTRY_NAMES[c]))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("Sentiment toward Ukraine")
    ax.set_ylabel("Sentiment toward Russia")
    ax.text(-1.0, 0.93, "more pro-Russian", fontsize=7.5, color="#444444", style="italic", va="top")
    ax.text(1.0, -0.93, "more pro-Ukrainian", fontsize=7.5, color="#444444", style="italic",
            ha="right", va="bottom")
    ax.text(0.62, 0.70, f"balanced band (|tilt| < {BALANCE_CUTOFF})", fontsize=7.5, color="#444444",
            rotation=45, ha="center", va="center")
    place_labels(ax, items)
    h = group_handles(groups) + [Line2D([], [], color="k", ls="--", lw=0.8, label="Equidistance (tilt = 0)")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 0.90), frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, 1, formats)


# ------------------------------------------------- fig 2: asymmetry over time
def fig2(d, formats, **_):
    a = d.dropna(subset=["asymmetry"])
    groups = present_groups(a)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ymax = 0
    for g in groups:
        m = a[a["group"] == g].groupby("t")["asymmetry"].mean()
        ymax = max(ymax, m.max())
        ax.plot(m.index, m.values, color=GROUP_COLOR[g], marker=GROUP_MARKER[g], ms=3.5, lw=1.1,
                label=GROUP_LABEL[g])
    inv_t = int(d.loc[d["quarter"] == INVASION_QUARTER, "t"].iloc[0])
    top = max(1.0, round(ymax * 1.15, 1))
    ax.axvline(inv_t, color="k", lw=0.7, ls=":")
    ax.text(inv_t + 0.2, top * 0.97, f"invasion\n({INVASION_QUARTER[-2:]} {INVASION_QUARTER[:4]})",
            fontsize=7.5, va="top")
    ax.axhline(BALANCE_CUTOFF, color="#999999", lw=0.6, ls="--")
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xticklabels(["2020", "2021", "2022", "2023", "2024"], fontsize=8)
    ax.set_xlim(-0.5, d["t"].max() + 0.5)
    ax.set_ylim(0, top)
    ax.set_ylabel("Asymmetry (absolute tilt)")
    ax.set_xlabel("Quarter")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, 2, formats)


# ------------------------------------------------ fig 3: joint-trust quadrant
def fig3(d, formats, **_):
    post = d[d["post"] == 1]
    cm = (post.groupby("country").agg(x=("recv_RUS", "mean"), y=("recv_UKR", "mean"))
              .dropna().assign(group=lambda t: t.index.map(GROUP_OF)))
    groups = present_groups(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    items = []
    rx = 0.05 * max(cm["x"].max() - cm["x"].min(), 0.05)
    ry = 0.05 * max(cm["y"].max() - cm["y"].min(), 0.05)
    for g in groups:
        s = cm[cm["group"] == g]
        cloud(ax, s[["x", "y"]].values, GROUP_COLOR[g], rx, ry, ls=GROUP_LINESTYLE[g])
        ax.scatter(s["x"], s["y"], s=26, color=GROUP_COLOR[g], marker=GROUP_MARKER[g],
                   edgecolor="black", linewidth=0.4, alpha=0.9, zorder=3)
        items += [(r["x"], r["y"], c) for c, r in s.iterrows()]
        ax.scatter([s["x"].mean()], [s["y"].mean()], s=260, color=GROUP_COLOR[g], marker=GROUP_MARKER[g],
                   edgecolor="black", linewidth=1.8, zorder=5)
    xl = (min(cm["x"].min(), 0) - 0.04, max(cm["x"].max(), 0) + 0.06)
    yl = (min(cm["y"].min(), 0) - 0.04, max(cm["y"].max(), 0) + 0.06)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.fill_between([0, xl[1]], 0, yl[1], color="#009E73", alpha=0.07, lw=0, zorder=0)
    ax.text(xl[1] - 0.005, yl[1] - 0.01, "joint-trust region\n(both sides positive)", ha="right",
            va="top", fontsize=6.8, style="italic", color="0.3")
    ax.axhline(0, color="0.4", ls="--", lw=0.6)
    ax.axvline(0, color="0.4", ls="--", lw=0.6)
    place_labels(ax, items, fontsize=6.5)
    ax.set_xlabel("Mean sentiment from Russia (post-invasion)")
    ax.set_ylabel("Mean sentiment from Ukraine (post-invasion)")
    handles = group_handles(groups) + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.7", markeredgecolor="black",
               markeredgewidth=1.3, ms=9, label="Large symbol = group mean;\noutlined cloud = group extent")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, 3, formats)


# ------------------------------------------------------ fig 4: tilt by group
def fig4(d, formats, demean=True, **_):
    t = d.dropna(subset=["tilt"]).copy()
    if demean:
        base = t[t["quarter"] < ANTICIPATION_START].groupby("country")["tilt"].mean()
        fallback = t[t["quarter"] < INVASION_QUARTER].groupby("country")["tilt"].mean()
        base = base.combine_first(fallback)
        missing = sorted(set(t["country"]) - set(base.index))
        if missing:
            print(f"  [warn] no pre-invasion tilt for {missing}; omitted from fig4")
            t = t[~t["country"].isin(missing)]
        t["y"] = t["tilt"] - t["country"].map(base)
        ylabel = "Tilt (pro-Russia \u2191), centred on\ncountry's baseline mean"
    else:
        t["y"] = t["tilt"]
        ylabel = "Tilt = sentiment to Russia \u2212 sentiment to Ukraine"

    groups = present_groups(t)
    fig, axes = plt.subplots(1, len(groups), figsize=(6.5, 2.7), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    ant_start = pd.Period(ANTICIPATION_START, freq="Q").start_time
    for ax, g in zip(axes, groups):
        sub = t[t["group"] == g]
        n = sub["country"].nunique()
        ends = []
        for c, s in sub.groupby("country"):
            s = s.sort_values("quarter_ts")
            ax.plot(s["quarter_ts"], s["y"], color=GROUP_COLOR[g], alpha=0.4, lw=0.8, zorder=2)
            ends.append((c, s["quarter_ts"].iloc[-1], s["y"].iloc[-1]))
        if n <= 3:
            adj = spread_labels([e[2] for e in ends], gap=0.075)
            for (c, xe, ye), ya in zip(ends, adj):
                ax.annotate(c, xy=(xe, ye), xytext=(xe + pd.Timedelta(days=70), ya), textcoords="data",
                            fontsize=6, color="0.25", va="center",
                            arrowprops=dict(arrowstyle="-", lw=0.4, color="0.55", shrinkA=0, shrinkB=1))
        m = sub.groupby("quarter_ts")["y"].mean().sort_index()
        ax.plot(m.index, m.values, color=GROUP_COLOR[g], lw=1.8, marker=GROUP_MARKER[g], ms=3.6,
                mec="black", mew=0.4, zorder=4)
        ax.axvspan(ant_start, WAR_DATE, color="0.85", lw=0, zorder=0)
        ax.axvline(WAR_DATE, color="black", ls="--", lw=0.8, zorder=1)
        ax.axhline(0, color="0.45", lw=0.5, zorder=1)
        ax.set_title(f"{GROUP_LABEL[g]}\n(n = {n})", fontsize=8)
        time_axis(ax)
    axes[0].set_ylabel(ylabel)
    axes[0].set_xlim(t["quarter_ts"].min() - pd.Timedelta(days=40),
                     t["quarter_ts"].max() + pd.Timedelta(days=170))
    handles = [
        Line2D([0], [0], color="0.3", lw=1.8, marker="o", ms=3.6, mec="black", mew=0.4, label="Group mean"),
        Line2D([0], [0], color="0.3", lw=0.8, alpha=0.5, label="Individual countries"),
        Patch(facecolor="0.85", label="Anticipation window (2021Q3\u2013Feb 2022)"),
        Line2D([0], [0], color="black", ls="--", lw=0.8, label="Invasion (24 Feb 2022)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, 4, formats)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figs", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    ap.add_argument("--formats", nargs="+", default=["pdf"])
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--no-demean", action="store_true", help="fig4: plot raw tilt")
    args = ap.parse_args()

    d = prep(load_panel(args.dedupe))
    funcs = {1: fig1, 2: fig2, 3: fig3, 4: fig4}
    for k in args.figs:
        print(f"fig{k}")
        funcs[k](d, args.formats, demean=not args.no_demean)


if __name__ == "__main__":
    main()