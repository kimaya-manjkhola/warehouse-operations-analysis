"""Project settings in one place.

Every assumption that changes a result (the delay threshold, the random seed,
the number of bootstrap repeats, the scenario sizes) lives here so it can be
seen, questioned and changed without touching the analysis code.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "raw_data.csv"
CLEAN_DATA = ROOT / "data" / "clean_data.csv"
OUTPUT_DIR = ROOT / "outputs"
PUBLISHED_CHARTS_DIR = ROOT / "charts"  # copy of the charts shown in the README
README = ROOT / "README.md"

# ---------------------------------------------------------------- assumptions
DELAY_THRESHOLD = 15.0  # minutes; an order is "delayed" if Processing Time > this
SENSITIVITY_THRESHOLDS = (12.0, 15.0, 18.0)
RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 500
CI_LEVEL = 0.95
ALPHA = 0.05
POWER = 0.80
TRAIN_SHARE = 0.80  # share of orders used to fit the out-of-sample check

# Trial design: a practical minimum length so every shift and weekday is seen
# several times (orders from one worker-shift are not independent observations)
TRIAL_MIN_DAYS = 14

# Scenario settings
S3_DELTAS = (0.5, 1.0, 1.5, 2.0)  # minutes removed from Zone C's base pick time
S4_DELTA = 1.0
MIN_PICK_TIME = 0.5  # a pick can never be faster than this after a scenario

# Data definitions
DATA_YEAR, DATA_MONTH = 2026, 8
SUM_TOLERANCE = 0.01  # allowed gap between Processing and Pick + Pack (rounding)
ZONES = ("A", "B", "C", "D")
SHIFTS = ("Morning", "Afternoon", "Evening")
EXPERIENCE = ("Experienced", "New")
ERROR_LABELS = ("Yes", "No")
WORKLOAD_BINS = (-float("inf"), 99, 129, float("inf"))
WORKLOAD_LABELS = ("<100", "100-129", "130+")
KEY_ZONE = "C"

# ---------------------------------------------------------------- known truth
# The process used to GENERATE the synthetic data (from the Excel README sheet).
# It is used ONLY to validate the fitted models, never as an input to them.
TRUE_PICK = {"base_A": 4.0, "base_B": 5.0, "base_C": 7.0, "base_D": 5.0,
             "k": 0.3, "m_evening": 1.15, "m_new": 1.25}
TRUE_PACK = {"p_new": 1.15}
TRUE_ERROR_PP = {"baseline": 2.0, "zone_C": 4.0, "evening": 2.0, "new_worker": 3.0}
TRUE_SHIFT_MIX = {"Morning": 0.30, "Afternoon": 0.30, "Evening": 0.40}

# ---------------------------------------------------------------- chart style
DPI = 150
COLOURS = {
    "highlight": "#eb6834",   # the key group (Zone C / new workers in Zone C)
    "base": "#86b6ef",        # every other group
    "dark": "#2a78d6",        # secondary series
    "neutral": "#c3c2b7",     # context series (e.g. share of orders)
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e4e3df",
    "surface": "#fcfcfb",
    "truth": "#0b0b0b",
}
SHIFT_COLOURS = {"Morning": "#2a78d6", "Afternoon": "#1baf7a", "Evening": "#eb6834"}
HEATMAP_CMAP = "Blues"
