#!/usr/bin/env python3
"""
02b_validation.py

Manual validation of the ABSA scores.

  python Scripts/02b_validation.py --make        # draw sample, write CSV to label
  (label Output/02b_validation_sample.csv in Excel/LibreOffice, column manual_label)
  python Scripts/02b_validation.py --evaluate    # agreement with the model

SAMPLE: unique (sentence, aspect) pairs, stratified by direction x belligerent
(RUS/UKR) x model label (Negative/Neutral/Positive) -> 12 strata, --per-stratum
rows each (default 20 -> ~240 rows). Stratifying by model label guarantees that
rare labels get checked; evaluation re-weights by stratum size, so reported
accuracy / kappa refer to the whole corpus, not to the sample.

BLIND LABELING: the CSV you label shows who speaks about whom and the sentence,
but NOT the model output (that is in 02b_validation_key.csv).

HOW TO LABEL (column manual_label). Judge ONLY the sentence shown, and the
sentiment expressed TOWARD the country in `about` (as named in aspect_term):
  positive  (pos, +, 1)     favourable / warm / supportive toward that country
  negative  (neg, -, -1)    critical / hostile / condemning toward that country
  neutral   (neu, 0)        factual, procedural, or no clear stance
  x                         not a real reference to the country (alias false
                            positive, e.g. "Indian Ocean")
  ?                         cannot decide -> excluded from evaluation
Leave `notes` for anything you want to remember. Partial labeling is fine:
--evaluate uses whatever is labeled so far.

Outputs (Output/):
  02b_validation_sample.csv   to label (never overwritten without --force)
  02b_validation_key.csv      model output, strata, weights
  02b_validation_report.txt   agreement metrics
  02b_validation_merged.csv   sample + key + normalized labels
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import OUTPUT_DIR

INPUT = OUTPUT_DIR / "02_mentions_absa.parquet"
SAMPLE_CSV = OUTPUT_DIR / "02b_validation_sample.csv"
KEY_CSV = OUTPUT_DIR / "02b_validation_key.csv"
REPORT_TXT = OUTPUT_DIR / "02b_validation_report.txt"
MERGED_CSV = OUTPUT_DIR / "02b_validation_merged.csv"

LABELS = ["Negative", "Neutral", "Positive"]  # ordinal order
VALUE = {"Negative": -1, "Neutral": 0, "Positive": 1}

LABEL_MAP = {
    "positive": "Positive", "pos": "Positive", "p": "Positive", "+": "Positive", "+1": "Positive", "1": "Positive",
    "negative": "Negative", "neg": "Negative", "-": "Negative", "-1": "Negative",
    "neutral": "Neutral", "neu": "Neutral", "0": "Neutral",
    "?": "SKIP", "skip": "SKIP", "unclear": "SKIP",
    "x": "NOT_COUNTRY", "wrong": "NOT_COUNTRY", "fp": "NOT_COUNTRY",
}


# --------------------------------------------------------------------------- #
# make
# --------------------------------------------------------------------------- #
def make(args):
    if SAMPLE_CSV.exists() and not args.force:
        raise SystemExit(f"{SAMPLE_CSV} already exists (it may contain your labels). "
                         f"Use --force to overwrite.")
    if not INPUT.exists():
        raise SystemExit(f"{INPUT} not found. Run 02_score_absa.py first.")

    df = pd.read_parquet(INPUT)
    need = {"pair_key", "sentence_text", "aspect_term", "absa_label", "absa_score"}
    if need - set(df.columns):
        raise SystemExit(f"Input lacks columns: {need - set(df.columns)}")
    df = df[df["absa_label"].notna()]

    pairs = df.drop_duplicates("pair_key").copy()
    pairs["belligerent"] = np.where(pairs["direction"] == "given", pairs["target_iso3"], pairs["source_iso3"])
    print(f"{len(pairs)} unique scored pairs")

    parts = []
    for keys, grp in pairs.groupby(["direction", "belligerent", "absa_label"]):
        n = min(args.per_stratum, len(grp))
        parts.append(grp.sample(n, random_state=args.seed)
                        .assign(stratum="|".join(keys), weight=len(grp) / n))
    samp = (pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True))
    samp["val_id"] = np.arange(1, len(samp) + 1)

    label_df = samp.assign(speaker=samp["source_iso3"], about=samp["target_iso3"],
                           manual_label="", notes="")[
        ["val_id", "speaker", "about", "aspect_term", "sentence_text", "url", "manual_label", "notes"]]
    key_df = samp[["val_id", "pair_key", "stratum", "weight", "direction", "belligerent",
                   "absa_label", "absa_prob_positive", "absa_prob_negative",
                   "absa_prob_neutral", "absa_score"]]

    label_df.to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")
    key_df.to_csv(KEY_CSV, index=False, encoding="utf-8-sig")
    print(f"Wrote {SAMPLE_CSV} ({len(label_df)} rows) and {KEY_CSV}\n"
          f"Label the column manual_label (see the docstring: positive / negative / neutral / x / ?).")


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def normalize(x):
    if pd.isna(x) or str(x).strip() == "":
        return None
    s = str(x).strip().lower()
    try:                       # Excel may turn "1" into 1.0, "+1" into 1
        f = float(s)
        s = str(int(f)) if f == int(f) else s
        if s.lstrip("-").isdigit() and int(s) > 0:
            s = "1"
    except ValueError:
        pass
    return LABEL_MAP.get(s, "INVALID")


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kappa(cm: np.ndarray, quadratic=False):
    n = cm.sum()
    k = cm.shape[0]
    if quadratic:
        i, j = np.indices(cm.shape)
        w = 1 - ((i - j) / (k - 1)) ** 2
    else:
        w = np.eye(k)
    po = (w * cm).sum() / n
    pe = (w * np.outer(cm.sum(1), cm.sum(0))).sum() / n ** 2
    return (po - pe) / (1 - pe)


def wmean(x, w):
    return float(np.average(x, weights=w)) if len(x) else np.nan


def evaluate(args):
    for p in (SAMPLE_CSV, KEY_CSV):
        if not p.exists():
            raise SystemExit(f"{p} not found. Run with --make first.")
    sample = pd.read_csv(SAMPLE_CSV, encoding="utf-8-sig", dtype={"manual_label": str})
    key = pd.read_csv(KEY_CSV, encoding="utf-8-sig")
    m = sample.merge(key, on="val_id", how="left")
    m["manual"] = m["manual_label"].map(normalize)

    invalid = m[m["manual"] == "INVALID"]
    if len(invalid):
        raise SystemExit("Unrecognised labels in val_id: " + ", ".join(map(str, invalid["val_id"])) +
                         "\n(accepted: positive/pos/+/1, negative/neg/-/-1, neutral/neu/0, x, ?)")

    labeled = m[m["manual"].notna()].copy()
    out = [f"02b_validation  labeled {len(labeled)} of {len(m)} rows"]
    if labeled.empty:
        raise SystemExit("No labels found yet.")

    # --- reference-identification and skips ------------------------------ #
    n_skip = int((labeled["manual"] == "SKIP").sum())
    n_fp = int((labeled["manual"] == "NOT_COUNTRY").sum())
    ref = labeled[labeled["manual"] != "SKIP"]
    fp_share_w = wmean((ref["manual"] == "NOT_COUNTRY").astype(float), ref["weight"])
    out += [f"skipped (?): {n_skip}",
            f"alias false positives (x): {n_fp} ({n_fp / len(ref):.1%} unweighted, {fp_share_w:.1%} weighted)"]
    if n_fp:
        fp = (ref[ref["manual"] == "NOT_COUNTRY"].groupby("aspect_term").size()
              .sort_values(ascending=False).head(10))
        out += ["  most frequent false-positive aspect terms:", fp.to_string()]

    ev = ref[ref["manual"].isin(LABELS)].copy()
    ev["manual_val"] = ev["manual"].map(VALUE)
    n = len(ev)
    out += ["", f"rows used for sentiment agreement: {n}", ""]
    if n < 10:
        raise SystemExit("\n".join(out) + "\nToo few labeled rows for metrics.")

    # --- confusion matrices ---------------------------------------------- #
    unw = pd.crosstab(ev["manual"], ev["absa_label"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
    wgt = (ev.pivot_table(index="manual", columns="absa_label", values="weight", aggfunc="sum")
             .reindex(index=LABELS, columns=LABELS).fillna(0.0))
    out += ["Confusion matrix, counts (rows = manual, cols = model):", unw.to_string(), "",
            "Confusion matrix, re-weighted to corpus (rows = manual, cols = model):",
            wgt.round(0).astype(int).to_string(), ""]

    correct = (ev["manual"] == ev["absa_label"]).astype(float)
    lo, hi = wilson(int(correct.sum()), n)
    out += [f"Accuracy (unweighted): {correct.mean():.1%}  [95% Wilson CI {lo:.1%}, {hi:.1%}]",
            f"Accuracy (weighted to corpus): {wmean(correct, ev['weight']):.1%}",
            f"Cohen's kappa (weighted data): {kappa(wgt.values):.3f}   quadratic-weighted kappa: {kappa(wgt.values, True):.3f}",
            f"Cohen's kappa (unweighted data): {kappa(unw.values.astype(float)):.3f}",
            f"Spearman rho, model absa_score vs manual (-1/0/1): {ev['absa_score'].corr(ev['manual_val'], method='spearman'):.3f}",
            ""]

    # per-class precision / recall / F1 (weighted confusion)
    cmw = wgt.values
    rows = []
    for i, lab in enumerate(LABELS):
        tp = cmw[i, i]
        prec = tp / cmw[:, i].sum() if cmw[:, i].sum() else np.nan
        rec = tp / cmw[i, :].sum() if cmw[i, :].sum() else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec + rec) else np.nan
        rows.append((lab, prec, rec, f1))
    cls = pd.DataFrame(rows, columns=["class", "precision", "recall", "F1"]).set_index("class").round(3)
    out += ["Per-class metrics (weighted):", cls.to_string(),
            f"macro-F1: {cls['F1'].mean():.3f}", ""]

    # --- bias by direction x belligerent (key concern for the paper) ----- #
    b = []
    for (d, bel), g in ev.groupby(["direction", "belligerent"]):
        b.append({"direction": d, "belligerent": bel, "n": len(g),
                  "model_mean": wmean(g["absa_score"], g["weight"]),
                  "manual_mean": wmean(g["manual_val"], g["weight"]),
                  "accuracy_w": wmean((g["manual"] == g["absa_label"]).astype(float), g["weight"])})
    bias = pd.DataFrame(b)
    bias["bias_model_minus_manual"] = bias["model_mean"] - bias["manual_mean"]
    out += ["Mean sentiment by direction x belligerent (weighted); bias = model - manual:",
            bias.round(3).to_string(index=False),
            "(manual_mean uses -1/0/+1, model_mean is P(pos)-P(neg); compare signs and ordering, "
            "not exact magnitudes)", ""]

    text = "\n".join(out)
    REPORT_TXT.write_text(text, encoding="utf-8")
    m.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")
    unw.to_csv(OUTPUT_DIR / "02b_validation_confusion.csv")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make", action="store_true")
    g.add_argument("--evaluate", action="store_true")
    ap.add_argument("--per-stratum", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite an existing sample file")
    args = ap.parse_args()
    make(args) if args.make else evaluate(args)


if __name__ == "__main__":
    main()