"""utils.py — shared helpers: panel loading, number formatting, LaTeX output."""

from __future__ import annotations

import re

import pandas as pd

from config import GROUPS, GROUP_OF, OUTPUT_DIR

COUNTRY_NAMES = {
    "BRA": "Brazil", "CHN": "China", "HUN": "Hungary", "IND": "India", "TUR": "Turkey",
    "AUT": "Austria", "CHE": "Switzerland", "IRL": "Ireland", "MLT": "Malta",
    "FIN": "Finland", "SWE": "Sweden", "RUS": "Russia", "UKR": "Ukraine",
}
GROUP_ORDER = list(GROUPS)                                         # diplomatic, military, former
COUNTRY_ORDER = [iso for g in GROUPS for iso in GROUPS[g]]
GROUP_SHORT = {"diplomatic": "Diplomatic", "military": "Military", "former": "Former"}
GROUP_CAP = {"diplomatic": "Dipl", "military": "Mil", "former": "For"}   # for LaTeX macro names


def load_panel(dedupe: bool = False) -> pd.DataFrame:
    path = OUTPUT_DIR / f"03_panel{'_dedup' if dedupe else ''}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run 03_build_panel.py first.")
    return pd.read_parquet(path)


# ---------------------------------------------------------------- numbers --
def fmt(x, nd=2, plus=False) -> str:
    """Table cell: negative numbers in math mode ($-0.05$), optional explicit plus."""
    if x is None or pd.isna(x):
        return "--"
    r = round(float(x), nd)
    s = f"{abs(r):.{nd}f}"
    if r < 0:
        return f"$-{s}$"
    return f"$+{s}$" if (r > 0 and plus) else s


def num(x, nd=2) -> str:
    """Macro-safe number (works in text and math mode)."""
    r = round(float(x), nd)
    s = f"{abs(r):.{nd}f}"
    return f"\\ensuremath{{-{s}}}" if r < 0 else s


def pct(x, nd=0) -> str:
    return "--" if x is None or pd.isna(x) else f"{100 * x:.{nd}f}\\%"


def pfmt(p) -> str:
    if p is None or pd.isna(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ LaTeX --
def latex_table(header, rows, caption, label, colfmt, note=None) -> str:
    """rows: list of lists of ready-made cell strings; the string 'midrule' inserts a rule."""
    out = [r"\begin{table}[htbp]", r"\centering", r"\small",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{@{{}}{colfmt}@{{}}}}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        out.append(r"\midrule" if r == "midrule" else " & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if note:
        out += [r"\begin{flushleft}", r"\footnotesize", note, r"\end{flushleft}"]
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def write_macros(d: dict, path) -> None:
    """Write \\newcommand-style macros (idempotent: providecommand + renewcommand)."""
    lines = ["% auto-generated, do not edit"]
    for k, v in d.items():
        if not re.fullmatch(r"[A-Za-z]+", k):
            raise ValueError(f"LaTeX macro names must be letters only: {k}")
        lines.append(rf"\providecommand{{\{k}}}{{}}\renewcommand{{\{k}}}{{{v}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")