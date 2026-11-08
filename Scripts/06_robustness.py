#!/usr/bin/env python3
"""
06_robustness.py — robustness checks and small-sample inference for H1-H4.

PARTS (run all, or choose with --parts):
  variants  re-estimate H1-H4 on altered samples / definitions (hypotheses.py functions):
              lowmention_5 / _10   exclude country-quarters with fewer than k mentions toward a
                                   belligerent (given side) or from a belligerent (received side)
              drop_2022Q1          drop the invasion quarter itself (mixed pre/post)
              pre_included_H1H4    H1/H4 with pre-invasion quarters included
              dedupe / no_dedupe   the other deduplication choice (each sentence counted once per
                                   source, target and direction vs. all mentions); needs the
                                   other 03 output, i.e. run 03_build_panel.py with or without --dedupe
              balance_cut_*        other "balanced" cutoffs (H1 balanced share)
              both_pos_thr_*       other thresholds for "positive" (H4 both-positive)
  loo       leave-one-country-out (11 runs) of everything
  boot      MENTION-LEVEL BOOTSTRAP: resample mentions with replacement within each
            country-quarter-belligerent cell, rebuild the group means, repeat B times.
            Shows how much sampling noise in the mentions alone moves the results, and
            estimates the upward bias of asymmetry (|noisy tilt| is biased upward).
            Captures within-cell noise only, NOT differences between countries.
  perm      EXACT RANDOMIZATION INFERENCE on country-level outcomes: reassign the group
            labels across countries in every possible way; p = share of assignments with a
            difference at least as large as the observed one. No distributional assumptions.
            (With 2 former neutrals vs 5 diplomatic there are only 21 assignments, so the
            smallest attainable p is 1/21; the table shows this floor.)
  wild      WILD CLUSTER BOOTSTRAP (null-imposed, Webb 6-point weights, clusters = countries)
            for the group contrasts of H1/H4, the DiD of H2 and the slope contrasts of H3.
            CAUTION: with only 2 "treated" clusters (former neutrals) wild bootstrap p-values
            are unreliable (MacKinnon & Webb 2017); prefer 'perm' for the former contrasts.
            NB sign: contrasts here are comparison MINUS diplomatic, also for H3 slopes.

Outputs (Output/):
  06_variants_long.csv, 06_variants_est.csv, 06_variants_p.csv
  06_loo_long.csv, 06_loo_est.csv, 06_loo_p.csv
  06_bootstrap.csv, 06_permutation.csv, 06_wild.csv
  06_summary.txt, 06_numbers.tex (leave-one-out ranges as macros, e.g. \\looMinhOneAsymMil)

Usage:
  python Scripts/06_robustness.py --dedupe                      # main specification: deduplicated sentences
  python Scripts/06_robustness.py --dedupe --parts perm wild --B-wild 9999
  (without --dedupe the baseline is the full, non-deduplicated panel)
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import time
import warnings
from functools import partial

import numpy as np
import pandas as pd
import patsy
from scipy import stats
from tqdm import tqdm

from config import (BALANCE_CUTOFF, GROUP_OF, GROUPS, INVASION_QUARTER, OUTPUT_DIR,
                    POSITIVE_THRESHOLD, RECEIVING_COUNTRIES, SAMPLE_END, SAMPLE_START,
                    SOURCE_COUNTRIES)
from hypotheses import h1, h2, h3, h4
from utils import load_panel, num, write_macros

MENTIONS = OUTPUT_DIR / "02_mentions_absa.parquet"
GROUPS3 = list(GROUPS)

KEYS = [
    "hOneAsymMil", "hOneAsymFor", "hOneBalMil", "hOneBalFor", "hOneValMil", "hOneValFor",
    "hTwoAsymWithinDipl", "hTwoAsymDidMil", "hTwoAsymDidFor", "hTwoTiltWithinDipl",
    "hThreeUkrDipl", "hThreeRusDipl", "hThreeUkrDiffMil", "hThreeUkrDiffFor",
    "hThreeRusDiffMil", "hThreeRusDiffFor",
    "hFourRecvMeanMil", "hFourRecvMeanFor", "hFourRecvMinMil", "hFourRecvMinFor",
    "hFourBothPosMil", "hFourBothPosFor",
]
GIVEN_COLS = ["score_RUS", "score_UKR", "tilt", "asymmetry", "valence", "balanced"]
RECV_COLS = ["recv_RUS", "recv_UKR", "recv_mean", "recv_min", "recv_both_pos"]


# =========================================================================== #
# 1. variants and leave-one-out
# =========================================================================== #
def run_variant(name, panel, fns=None):
    fns = fns or [h1, h2, h3, h4]
    rows = []
    for fn in fns:
        try:
            rows += fn(panel)
        except Exception as e:  # keep going; report
            print(f"  [{name}] {getattr(fn, '__name__', 'fn')} failed: {type(e).__name__}: {e}")
    d = pd.DataFrame(rows)
    if len(d):
        d.insert(0, "variant", name)
    return d


def mask_low(panel, k):
    p = panel.copy()
    low_g = p[["n_RUS", "n_UKR"]].min(axis=1) < k
    low_r = p[["n_recv_RUS", "n_recv_UKR"]].min(axis=1) < k
    p.loc[low_g, GIVEN_COLS] = np.nan
    p.loc[low_r, RECV_COLS] = np.nan
    return p


def with_cutoff(panel, cut):
    p = panel.copy()
    p["balanced"] = np.where(p["asymmetry"].notna(), (p["asymmetry"] < cut).astype(float), np.nan)
    return p


def with_pos_thr(panel, thr):
    p = panel.copy()
    both = p["recv_RUS"].notna() & p["recv_UKR"].notna()
    p["recv_both_pos"] = np.where(both, ((p["recv_RUS"] > thr) & (p["recv_UKR"] > thr)).astype(float), np.nan)
    return p


def part_variants(panel, dedupe=False):
    out = [run_variant("baseline", panel)]
    out.append(run_variant("lowmention_5", mask_low(panel, 5)))
    out.append(run_variant("lowmention_10", mask_low(panel, 10)))
    out.append(run_variant("drop_2022Q1", panel[panel["quarter"] != INVASION_QUARTER]))
    out.append(run_variant("pre_included_H1H4", panel,
                           [partial(h1, post_only=False), partial(h4, post_only=False)]))
    # the other deduplication choice as a variant (requires the other 03 output to exist)
    other = "no_dedupe" if dedupe else "dedupe"
    try:
        out.append(run_variant(other, load_panel(dedupe=not dedupe)))
    except SystemExit:
        print(f"  (variant '{other}' skipped: run 03_build_panel.py"
              f"{'' if dedupe else ' --dedupe'} to create the other panel)")
    for c in (0.15, 0.20, 0.30, 0.35):
        out.append(run_variant(f"balance_cut_{c:.2f}", with_cutoff(panel, c), [h1]))
    for t in (0.05, 0.10):
        out.append(run_variant(f"both_pos_thr_{t:.2f}", with_pos_thr(panel, t), [h4]))
    return pd.concat([o for o in out if len(o)], ignore_index=True)

def part_loo(panel):
    out = [run_variant("baseline", panel)]
    for iso in tqdm(SOURCE_COUNTRIES, desc="leave-one-out"):
        out.append(run_variant(f"drop_{iso}", panel[panel["country"] != iso]))
    return pd.concat([o for o in out if len(o)], ignore_index=True)


def wide(df, col, order):
    d = df[df["key"].isin(KEYS)]
    w = d.pivot_table(index="variant", columns="key", values=col, aggfunc="first")
    return w.reindex(index=[v for v in order if v in w.index],
                     columns=[k for k in KEYS if k in w.columns])


# =========================================================================== #
# 2. mention-level bootstrap
# =========================================================================== #
def part_boot(B, seed, cutoff, thr, dedupe):
    if not MENTIONS.exists():
        print(f"  {MENTIONS} missing; skipping bootstrap")
        return None
    parse_quarters = importlib.import_module("03_build_panel").parse_quarters
    rng = np.random.default_rng(seed)

    m = pd.read_parquet(MENTIONS, columns=["direction", "source_iso3", "target_iso3", "date", "absa_score", "sentence_text"])
    m["quarter"] = parse_quarters(m["date"])
    q0, q1 = pd.Period(SAMPLE_START, "Q"), pd.Period(SAMPLE_END, "Q")
    m = m[m["quarter"].notna() & m["absa_score"].notna()]
    m = m[(m["quarter"] >= q0) & (m["quarter"] <= q1)]
    g = (m["direction"] == "given") & m["source_iso3"].isin(SOURCE_COUNTRIES) & m["target_iso3"].isin(RECEIVING_COUNTRIES)
    r = (m["direction"] == "received") & m["source_iso3"].isin(RECEIVING_COUNTRIES) & m["target_iso3"].isin(SOURCE_COUNTRIES)
    m = m[g | r].reset_index(drop=True)
    if dedupe:
        m = m.drop_duplicates(["direction", "source_iso3", "target_iso3", "sentence_text"]).reset_index(drop=True)

    nC = len(SOURCE_COUNTRIES)
    nQ = (q1.year * 4 + q1.quarter) - (q0.year * 4 + q0.quarter) + 1
    inv = pd.Period(INVASION_QUARTER, "Q")
    inv_idx = (inv.year * 4 + inv.quarter) - (q0.year * 4 + q0.quarter)

    side = (m["direction"] == "received").astype(int).to_numpy()
    country = pd.Series(np.where(side == 0, m["source_iso3"], m["target_iso3"]))
    bel = pd.Series(np.where(side == 0, m["target_iso3"], m["source_iso3"]))
    cidx = country.map({c: i for i, c in enumerate(SOURCE_COUNTRIES)}).to_numpy()
    belc = (bel == "UKR").astype(int).to_numpy()
    qidx = ((m["quarter"].dt.year * 4 + m["quarter"].dt.quarter) - (q0.year * 4 + q0.quarter)).to_numpy()
    score = m["absa_score"].to_numpy(float)

    n_cells = 2 * nC * nQ * 2
    cell = (((side * nC + cidx) * nQ + qidx) * 2 + belc).astype(np.int64)
    order = np.argsort(cell, kind="stable")
    cell_s, score_s = cell[order], score[order]
    counts = np.bincount(cell_s, minlength=n_cells)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    cnt_row, start_row = counts[cell_s], starts[cell_s]

    def means_of(vals):
        sums = np.bincount(cell_s, weights=vals, minlength=n_cells)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, sums / counts, np.nan)

    country_group = np.array([GROUP_OF[c] for c in SOURCE_COUNTRIES])
    post_q = np.arange(nQ) >= inv_idx
    MEAS = ["asymmetry", "balanced", "valence", "recv_mean", "recv_min", "recv_both_pos"]
    LABS = ["diplomatic", "military", "former", "military - diplomatic", "former - diplomatic"]

    def vec(means):
        M = means.reshape(2, nC, nQ, 2)
        Gm, Rm = M[0], M[1]
        tilt = Gm[..., 0] - Gm[..., 1]
        asym = np.abs(tilt)
        arrs = {
            "asymmetry": asym,
            "balanced": np.where(np.isnan(asym), np.nan, (asym < cutoff).astype(float)),
            "valence": (Gm[..., 0] + Gm[..., 1]) / 2,
            "recv_mean": (Rm[..., 0] + Rm[..., 1]) / 2,
            "recv_min": np.minimum(Rm[..., 0], Rm[..., 1]),
            "recv_both_pos": np.where(np.isnan(Rm[..., 0]) | np.isnan(Rm[..., 1]), np.nan,
                                      ((Rm[..., 0] > thr) & (Rm[..., 1] > thr)).astype(float)),
        }
        out = []
        for k in MEAS:
            gm = {gr: np.nanmean(arrs[k][country_group == gr][:, post_q]) for gr in GROUPS3}
            out += [gm["diplomatic"], gm["military"], gm["former"],
                    gm["military"] - gm["diplomatic"], gm["former"] - gm["diplomatic"]]
        return np.array(out)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        point = vec(means_of(score_s))
        draws = np.empty((B, len(point)))
        for b in tqdm(range(B), desc="mention bootstrap"):
            idx = start_row + np.floor(rng.random(len(cell_s)) * cnt_row).astype(np.int64)
            draws[b] = vec(means_of(score_s[idx]))

    names = [(k, lab) for k in MEAS for lab in LABS]
    bm = np.nanmean(draws, axis=0)
    tab = pd.DataFrame({
        "measure": [n[0] for n in names], "quantity": [n[1] for n in names],
        "point": point, "boot_mean": bm, "boot_se": np.nanstd(draws, axis=0, ddof=1),
        "ci_lo": np.nanpercentile(draws, 2.5, axis=0), "ci_hi": np.nanpercentile(draws, 97.5, axis=0),
    })
    tab["bias"] = tab["boot_mean"] - tab["point"]
    tab["bias_corrected"] = tab["point"] - tab["bias"]
    return tab


# =========================================================================== #
# 3. exact randomization inference (country-level)
# =========================================================================== #
def exact_perm(vals: pd.Series, comp_groups):
    v = vals.dropna()
    grp = pd.Series({c: GROUP_OF[c] for c in v.index})
    A, Bv = v[grp == "diplomatic"], v[grp.isin(comp_groups)]
    if len(A) < 1 or len(Bv) < 1:
        return None
    pool = np.concatenate([A.to_numpy(), Bv.to_numpy()])
    N, nA, tot = len(pool), len(A), pool.sum()
    obs = Bv.mean() - A.mean()  # comparison minus diplomatic
    diffs = []
    for comb in itertools.combinations(range(N), nA):
        sA = pool[list(comb)].sum()
        diffs.append((tot - sA) / (N - nA) - sA / nA)
    diffs = np.array(diffs)
    return obs, len(diffs), float((np.abs(diffs) >= abs(obs) - 1e-12).mean()), 1 / len(diffs)


def part_perm(panel):
    post, pre = panel[panel["post"] == 1], panel[panel["post"] == 0]
    meas = ["asymmetry", "balanced", "valence", "recv_mean", "recv_min", "recv_both_pos"]
    level = post.groupby("country")[meas].mean()
    chg_cols = ["asymmetry", "tilt"]
    change = post.groupby("country")[chg_cols].mean() - pre.groupby("country")[chg_cols].mean()
    rows = []
    for hyp, tbl, cols in [("H1/H4 (post level)", level, meas), ("H2 (post - pre change)", change, chg_cols)]:
        for y in cols:
            for lab, comp in [("military - diplomatic", ["military"]), ("former - diplomatic", ["former"]),
                              ("(military+former) - diplomatic", ["military", "former"])]:
                res = exact_perm(tbl[y], comp)
                if res:
                    obs, n_perm, p, floor = res
                    rows.append(dict(analysis=hyp, outcome=y, contrast=lab, observed=obs,
                                     n_assignments=n_perm, p_exact=p, p_floor=floor))
    return pd.DataFrame(rows)


# =========================================================================== #
# 4. wild cluster bootstrap (null-imposed, Webb weights)
# =========================================================================== #
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def wild_cluster_p(data, formula, term, rng, B, cluster="country"):
    y, X = patsy.dmatrices(formula, data, return_type="dataframe")
    cl = data.loc[X.index, cluster].to_numpy()
    yv, Xv = y.iloc[:, 0].to_numpy(), X.to_numpy()
    cols = list(X.columns)
    j = cols.index(term)
    n, k = Xv.shape
    A = np.linalg.pinv(Xv.T @ Xv) @ Xv.T                       # k x n
    codes, uniq = pd.factorize(cl)
    G = len(uniq)
    D = np.zeros((n, G))
    D[np.arange(n), codes] = 1.0
    c = G / (G - 1) * (n - 1) / (n - k)                        # CR1 small-sample factor
    aj = A[j]
    beta = A @ yv
    u = yv - Xv @ beta
    se = np.sqrt(c * (((aj * u) @ D) ** 2).sum())
    t_hat = beta[j] / se

    keep = [i for i in range(k) if i != j]                     # restricted model (null imposed)
    Xr = Xv[:, keep]
    fit_r = Xr @ (np.linalg.pinv(Xr) @ yv)
    res_r = yv - fit_r
    W = rng.choice(WEBB, size=(G, B))
    Ystar = fit_r[:, None] + res_r[:, None] * W[codes, :]
    Bet = A @ Ystar
    U = Ystar - Xv @ Bet
    S = (aj[:, None] * U).T @ D
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = Bet[j] / np.sqrt(c * (S ** 2).sum(1))
    return dict(est=float(beta[j]), se_cr1=float(se), t=float(t_hat),
                p_cr1=float(2 * stats.t.sf(abs(t_hat), G - 1)),
                p_wild=float((np.abs(t_star) >= abs(t_hat)).mean()), n=n, G=G)


def part_wild(panel, B, seed):
    rng = np.random.default_rng(seed)
    d = panel.copy()
    d["mil"] = (d["group"] == "military").astype(int)
    d["form"] = (d["group"] == "former").astype(int)
    for g, s in [("diplomatic", "dipl"), ("military", "mil"), ("former", "form")]:
        d[f"post_{s}"] = ((d["post"] == 1) & (d["group"] == g)).astype(int)
    d["tilt_mil"] = d["tilt"] * d["mil"]
    d["tilt_form"] = d["tilt"] * d["form"]
    post = d[d["post"] == 1]
    rows = []

    def add(hyp, outcome, data, formula, term):
        try:
            rows.append(dict(hyp=hyp, outcome=outcome, term=term,
                             **wild_cluster_p(data, formula, term, rng, B)))
        except Exception as e:
            print(f"  wild {hyp} {outcome} {term} failed: {type(e).__name__}: {e}")

    for y in ["asymmetry", "balanced", "valence"]:
        for term in ("mil", "form"):
            add("H1", y, post, f"{y} ~ mil + form + C(quarter)", term)
    for y in ["recv_mean", "recv_min", "recv_both_pos"]:
        for term in ("mil", "form"):
            add("H4", y, post, f"{y} ~ mil + form + C(quarter)", term)
    for y in ["asymmetry", "tilt"]:
        for term in ("post_mil", "post_form"):
            add("H2", y, d, f"{y} ~ post_mil + post_form + C(country) + C(quarter)", term)
        add("H2", y, d, f"{y} ~ post_dipl + post_mil + post_form + C(country)", "post_dipl")
    for y in ["recv_UKR", "recv_RUS"]:
        dd = d[d["tilt"].notna() & d[y].notna()]
        for term in ("tilt", "tilt_mil", "tilt_form"):
            add("H3", y, dd, f"{y} ~ tilt + tilt_mil + tilt_form + C(country) + C(quarter)", term)
    return pd.DataFrame(rows)


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", default=["variants", "loo", "boot", "perm", "wild"],
                    choices=["variants", "loo", "boot", "perm", "wild"])
    ap.add_argument("--B-boot", type=int, default=1000)
    ap.add_argument("--B-wild", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_colwidth", 40)
    t0 = time.time()
    sfx = "_dedup" if args.dedupe else ""
    panel = load_panel(args.dedupe)
    txt = [f"06_robustness  ({time.strftime('%Y-%m-%d %H:%M:%S')})  parts: {', '.join(args.parts)}",
           "Estimates: OLS as in 05_models.py; p-values HC1 normal-approximation unless stated.", ""]

    if "variants" in args.parts:
        v = part_variants(panel, args.dedupe)
        order = list(dict.fromkeys(v["variant"]))
        v.to_csv(OUTPUT_DIR / f"06_variants_long{sfx}.csv", index=False)
        we, wp = wide(v, "est", order), wide(v, "p", order)
        we.to_csv(OUTPUT_DIR / f"06_variants_est{sfx}.csv")
        wp.to_csv(OUTPUT_DIR / f"06_variants_p{sfx}.csv")
        txt += ["=== VARIANTS: key estimates ===", we.round(3).T.to_string(), "",
                "=== VARIANTS: p-values ===", wp.round(3).T.to_string(), ""]

    if "loo" in args.parts:
        l = part_loo(panel)
        order = list(dict.fromkeys(l["variant"]))
        l.to_csv(OUTPUT_DIR / f"06_loo_long{sfx}.csv", index=False)
        we, wp = wide(l, "est", order), wide(l, "p", order)
        we.to_csv(OUTPUT_DIR / f"06_loo_est{sfx}.csv")
        wp.to_csv(OUTPUT_DIR / f"06_loo_p{sfx}.csv")
        rng_tab = pd.DataFrame({"baseline": we.loc["baseline"], "loo_min": we.drop("baseline").min(),
                                "loo_max": we.drop("baseline").max(),
                                "n_p<0.05": (wp.drop("baseline") < 0.05).sum()})
        txt += ["=== LEAVE-ONE-OUT: estimates (rows = dropped country) ===", we.round(3).T.to_string(), "",
                "=== LEAVE-ONE-OUT: p-values ===", wp.round(3).T.to_string(), "",
                "=== LEAVE-ONE-OUT: range of estimates ===", rng_tab.round(3).to_string(), ""]
        macros = {}
        for k in rng_tab.index:
            macros[f"looMin{k}"] = num(rng_tab.loc[k, "loo_min"])
            macros[f"looMax{k}"] = num(rng_tab.loc[k, "loo_max"])
        write_macros(macros, OUTPUT_DIR / f"06_numbers{sfx}.tex")

    if "boot" in args.parts:
        b = part_boot(args.B_boot, args.seed, BALANCE_CUTOFF, POSITIVE_THRESHOLD, args.dedupe)
        if b is not None:
            b.to_csv(OUTPUT_DIR / f"06_bootstrap{sfx}.csv", index=False)
            txt += [f"=== MENTION-LEVEL BOOTSTRAP (B={args.B_boot}; post-invasion group means) ===",
                    "bias = bootstrap mean - point estimate; bias_corrected = point - bias",
                    "(for asymmetry a positive bias means the point estimate is inflated by sampling noise)",
                    b.round(3).to_string(index=False), ""]

    if "perm" in args.parts:
        p = part_perm(panel)
        p.to_csv(OUTPUT_DIR / f"06_permutation{sfx}.csv", index=False)
        txt += ["=== EXACT RANDOMIZATION INFERENCE (country-level; contrast = comparison - diplomatic) ===",
                "p_floor = smallest attainable p given the number of possible label assignments",
                p.round(4).to_string(index=False), ""]

    if "wild" in args.parts:
        w = part_wild(panel, args.B_wild, args.seed)
        w.to_csv(OUTPUT_DIR / f"06_wild{sfx}.csv", index=False)
        txt += [f"=== WILD CLUSTER BOOTSTRAP (B={args.B_wild}, Webb weights, clusters = countries) ===",
                "p_cr1 = analytic cluster-robust p (t with G-1 df); p_wild = bootstrap-t p. "
                "H3: tilt = diplomatic slope; tilt_mil / tilt_form = slope minus diplomatic slope.",
                w.round(4).to_string(index=False), ""]

    txt.append(f"runtime: {time.time() - t0:.0f}s")
    text = "\n".join(txt)
    (OUTPUT_DIR / f"06_summary{sfx}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()