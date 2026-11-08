"""
hypotheses.py — estimation functions for H1-H4. Each takes the panel
(Output/03_panel*.parquet as DataFrame) and returns a list of result rows
(dicts). 05_models.py runs them on the main panel; 06_robustness.py reuses
them on altered panels (subsamples, leave-one-out, dedupe, ...).

Conventions
  * OLS with HC1 standard errors (paper), country/quarter fixed effects as dummies.
  * Group contrasts are "comparison group MINUS diplomatic neutrals".
  * Normal-approximation p-values (statsmodels default for robust covariances).
    With 10-11 countries these are too optimistic; see 06 for resampling-based inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from utils import GROUP_CAP

REF = "diplomatic"
GRP = f'C(group, Treatment("{REF}"))'
GROUPS3 = ["diplomatic", "military", "former"]


# ------------------------------------------------------------- plumbing --
def _fit(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="HC1")


def _find(res, *subs):
    hits = [n for n in res.params.index if all(s in n for s in subs)]
    if len(hits) != 1:
        raise KeyError(f"term {subs} matched {hits}")
    return hits[0]


def _mk(hyp, spec, outcome, term, est, se, p, lo, hi, n, key=None, note=""):
    return dict(hyp=hyp, spec=spec, outcome=outcome, term=term, est=est, se=se, p=p,
                lo=lo, hi=hi, n=n, key=key, note=note)


def _row(res, pname, hyp, spec, outcome, label, key=None, note=""):
    lo, hi = res.conf_int().loc[pname]
    return _mk(hyp, spec, outcome, label, float(res.params[pname]), float(res.bse[pname]),
               float(res.pvalues[pname]), float(lo), float(hi), int(res.nobs), key, note)


def _wald_joint(res, pnames):
    idx = [list(res.params.index).index(p) for p in pnames]
    R = np.zeros((len(idx), len(res.params)))
    for i, j in enumerate(idx):
        R[i, j] = 1.0
    try:
        w = res.wald_test(R, scalar=True)
    except TypeError:
        w = res.wald_test(R)
    return float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue))


def _contrast(res, plus, minus):
    names = list(res.params.index)
    R = np.zeros(len(names))
    R[names.index(plus)] = 1.0
    R[names.index(minus)] = -1.0
    t = res.t_test(R)
    return float(np.squeeze(t.effect)), float(np.squeeze(t.sd)), float(np.squeeze(t.pvalue))


def _tost_p(est, se, margin):
    """Two one-sided tests: H0 true effect <= -margin, and H0 true effect >= +margin."""
    p_lo = 1 - stats.norm.cdf((est + margin) / se)
    p_hi = stats.norm.cdf((est - margin) / se)
    return float(max(p_lo, p_hi))


# ----------------------------------------------------- group contrasts --
def group_contrasts(d, y, hyp, keybase):
    d = d[d[y].notna()]
    spec = f"{y} ~ group + quarter FE (post-invasion quarters)"
    res = _fit(f"{y} ~ {GRP} + C(quarter)", d)
    names = {g: _find(res, f"[T.{g}]") for g in ("military", "former")}
    rows = [_row(res, names[g], hyp, spec, y, f"{g} - diplomatic", key=f"{keybase}{GROUP_CAP[g]}")
            for g in names]
    stat, p = _wald_joint(res, list(names.values()))
    rows.append(_mk(hyp, spec, y, "joint: groups equal", np.nan, np.nan, p, np.nan, np.nan,
                    int(res.nobs), key=f"{keybase}Joint", note=f"chi2={stat:.2f}"))
    return rows


def h1(panel, post_only=True):
    """H1 (reality): post-invasion asymmetry / balanced share / valence by group."""
    d = panel[panel["post"] == 1] if post_only else panel
    rows = []
    for y, k in [("asymmetry", "Asym"), ("balanced", "Bal"), ("valence", "Val")]:
        rows += group_contrasts(d, y, "H1", f"hOne{k}")
    return rows


def h4(panel, post_only=True):
    """H4 (joint reception): post-invasion received mean / minimum / both-positive by group."""
    d = panel[panel["post"] == 1] if post_only else panel
    rows = []
    for y, k in [("recv_mean", "RecvMean"), ("recv_min", "RecvMin"), ("recv_both_pos", "BothPos")]:
        rows += group_contrasts(d, y, "H4", f"hFour{k}")
    return rows


# ------------------------------------------------------------------ H2 --
def h2(panel, margin_asym=0.15, margin_tilt=0.20):
    """
    H2 (invariance), difference-in-differences around the invasion (all quarters).
      (a) y ~ post_mil + post_form + country FE + quarter FE
          -> change of military / former RELATIVE to diplomatic neutrals
             (the diplomatic change itself is absorbed by quarter FE)
      (b) y ~ post_dipl + post_mil + post_form + country FE   (no quarter FE)
          -> each group's own pre->post change; TOST equivalence test on the
             diplomatic change against +-margin
    """
    d = panel.copy()
    for g, s in [("diplomatic", "dipl"), ("military", "mil"), ("former", "form")]:
        d[f"post_{s}"] = ((d["post"] == 1) & (d["group"] == g)).astype(int)
    rows = []
    for y, K, margin in [("asymmetry", "Asym", margin_asym), ("tilt", "Tilt", margin_tilt)]:
        dd = d[d[y].notna()]

        res = _fit(f"{y} ~ post_mil + post_form + C(country) + C(quarter)", dd)
        spec = f"{y} ~ post x group (DiD vs diplomatic) + country FE + quarter FE"
        for g, s in [("military", "post_mil"), ("former", "post_form")]:
            rows.append(_row(res, s, "H2", spec, y, f"{g} - diplomatic (change)",
                             key=f"hTwo{K}Did{GROUP_CAP[g]}"))

        res = _fit(f"{y} ~ post_dipl + post_mil + post_form + C(country)", dd)
        spec = f"{y} ~ group-specific post + country FE (within-group change)"
        for g, s in [("diplomatic", "post_dipl"), ("military", "post_mil"), ("former", "post_form")]:
            rows.append(_row(res, s, "H2", spec, y, f"{g}: change post vs pre",
                             key=f"hTwo{K}Within{GROUP_CAP[g]}"))
        est, se = float(res.params["post_dipl"]), float(res.bse["post_dipl"])
        p = _tost_p(est, se, margin)
        rows.append(_mk("H2", spec + f"; TOST +-{margin}", y, f"diplomatic: equivalence within +-{margin}",
                        est, se, p, est - 1.645 * se, est + 1.645 * se, int(res.nobs),
                        key=f"hTwo{K}Tost",
                        note="equivalent" if p < 0.05 else "equivalence NOT established (90% CI shown)"))
    return rows


# ------------------------------------------------------------------ H3 --
def h3(panel, sample="all"):
    """
    H3 (reciprocity): received sentiment from each belligerent ~ tilt, group-specific
    slopes, country FE + quarter FE. sample: "all" quarters or "post" only.
    Contrasts are diplomatic slope MINUS comparison slope.
    """
    d = panel if sample == "all" else panel[panel["post"] == 1]
    rows = []
    for bel, B in [("UKR", "Ukr"), ("RUS", "Rus")]:
        y = f"recv_{bel}"
        dd = d[d["tilt"].notna() & d[y].notna()]
        res = _fit(f"{y} ~ C(group):tilt + C(country) + C(quarter)", dd)
        spec = f"{y} ~ tilt x group + country FE + quarter FE ({sample} quarters)"
        pn = {g: _find(res, f"[{g}]", "tilt") for g in GROUPS3 if f"{g}" in set(dd["group"])}
        for g, p in pn.items():
            rows.append(_row(res, p, "H3", spec, y, f"slope:{g}", key=f"hThree{B}{GROUP_CAP[g]}"))
        if "diplomatic" in pn:
            for g in ("military", "former"):
                if g in pn:
                    est, se, p = _contrast(res, pn["diplomatic"], pn[g])
                    rows.append(_mk("H3", spec, y, f"slope diff: diplomatic - {g}", est, se, p,
                                    est - 1.96 * se, est + 1.96 * se, int(res.nobs),
                                    key=f"hThree{B}Diff{GROUP_CAP[g]}"))
        # diplomatic neutrals only
        dn = dd[dd["group"] == "diplomatic"]
        if len(dn) > 10:
            r2 = _fit(f"{y} ~ tilt + C(country) + C(quarter)", dn)
            rows.append(_row(r2, "tilt", "H3", f"{y} ~ tilt + country FE + quarter FE (diplomatic only)",
                             y, "slope-only:diplomatic", key=f"hThree{B}Only"))
    return rows