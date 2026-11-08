"""
config.py — shared constants for the GlobalDiplomacyNet pipeline.
Import with:  from config import ...
(Scripts in Scripts/ can import this directly when run as `python Scripts/xx.py`.)
"""

from pathlib import Path

# ----------------------------------------------------------------------- #
# Paths
# ----------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
OUTPUT_DIR = ROOT / "Output"
FIGURES_DIR = ROOT / "Graphs"
ALIASES_JSON = DATA_DIR / "country_aliases.json"

# ----------------------------------------------------------------------- #
# Countries and groups
# ----------------------------------------------------------------------- #
# Source countries by neutrality group. Edit here, nowhere else.
# NOTE: Malta is assigned to "military" (constitutionally neutral).
GROUPS = {
    "diplomatic": ["BRA", "CHN", "HUN", "IND", "TUR"],
    "military":   ["AUT", "CHE", "IRL", "MLT"],
    "former":     ["FIN", "SWE"],
}
GROUP_LABELS = {
    "diplomatic": "Diplomatic neutrals",
    "military": "Military neutrals",
    "former": "Former neutrals",
}
GROUP_OF = {iso: g for g, isos in GROUPS.items() for iso in isos}
SOURCE_COUNTRIES = sorted(GROUP_OF)
RECEIVING_COUNTRIES = ["RUS", "UKR"]  # the two belligerents

# ----------------------------------------------------------------------- #
# Time
# ----------------------------------------------------------------------- #
SAMPLE_START = "2020Q1"
SAMPLE_END = "2024Q4"
INVASION_QUARTER = "2022Q1"   # first "post" quarter

# ----------------------------------------------------------------------- #
# Measurement thresholds
# ----------------------------------------------------------------------- #
BALANCE_CUTOFF = 0.25      # asymmetry below this = "balanced" (diplomatic neutrality)
MIN_MENTIONS = 5           # country-quarter flagged "low_mention" if fewer mentions toward a belligerent
POSITIVE_THRESHOLD = 0.0   # received score above this counts as "positive"