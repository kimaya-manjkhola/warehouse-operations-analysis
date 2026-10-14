"""Statistical tests used to check that the patterns are not just noise.

* Wilson confidence intervals for rates (better than "p +/- 1.96*SE" when a
  rate is near 0% or 100%, or the group is small).
* Chi-square tests of independence with Cramer's V as the effect size
  (the p-value says "is there a link?", Cramer's V says "how strong is it?").
* Welch's t-test (means) and Mann-Whitney U (whole distributions) for
  processing times, because the times are skewed.
* Bonferroni correction, because running several tests raises the chance
  that at least one looks significant by luck.
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def wilson_ci(successes, n, level: float = config.CI_LEVEL):
    """Wilson score interval for a proportion successes / n.

    Works on single numbers or arrays. Returns (lower, upper) as proportions.
    """
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    z = stats.norm.ppf(0.5 + level / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def add_wilson_ci(table: pd.DataFrame, events: str = "delayed_orders",
                  n: str = "orders", prefix: str = "delay_rate") -> pd.DataFrame:
    """Return a copy of a summary table with Wilson CI columns for a rate."""
    out = table.copy()
    low, high = wilson_ci(out[events], out[n])
    out[f"{prefix}_ci_low"], out[f"{prefix}_ci_high"] = low, high
    return out


def cramers_v(chi2: float, n: int, shape: tuple) -> float:
    """Cramer's V: chi-square rescaled to 0 (no link) .. 1 (perfect link).

    Rough guide for a 2-column table: 0.1 small, 0.3 medium, 0.5 large.
    """
    k = min(shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def chi_square_test(df: pd.DataFrame, row: str, col: str, label: str) -> dict:
    """Chi-square test of independence between two categorical columns.

    Returns the statistic, degrees of freedom, p-value and Cramer's V.
    No continuity correction is used so V is comparable across table sizes.
    """
    table = pd.crosstab(df[row], df[col])
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return {"test": label, "method": "Chi-square", "statistic": float(chi2), "dof": int(dof),
            "p_value": float(p), "effect_size": cramers_v(chi2, int(table.values.sum()), table.shape),
            "effect_measure": "Cramer's V", "n": int(table.values.sum())}


def compare_times(a: pd.Series, b: pd.Series, label: str) -> list[dict]:
    """Compare two groups' processing times with Welch's t-test and Mann-Whitney U.

    Welch's t-test compares means without assuming equal variances. Mann-Whitney
    compares whole distributions and is robust to skew. Effect sizes: Cohen's d
    (difference in means in standard deviations) and the probability that a
    random order from group a takes longer than one from group b.
    """
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    base = {"test": label, "dof": np.nan, "n": int(len(a) + len(b))}
    return [
        {**base, "method": "Welch t-test", "statistic": float(t.statistic), "p_value": float(t.pvalue),
         "effect_size": float((a.mean() - b.mean()) / pooled_sd), "effect_measure": "Cohen's d"},
        {**base, "method": "Mann-Whitney U", "statistic": float(u.statistic), "p_value": float(u.pvalue),
         "effect_size": float(u.statistic / (len(a) * len(b))), "effect_measure": "P(a > b)"},
    ]


def bonferroni(p_values) -> np.ndarray:
    """Bonferroni-adjusted p-values: multiply by the number of tests, cap at 1."""
    p = np.asarray(p_values, dtype=float)
    return np.minimum(p * len(p), 1.0)


def run_test_suite(df: pd.DataFrame) -> pd.DataFrame:
    """Run the root-cause test family and apply a Bonferroni correction.

    The family: delay vs zone; delay vs experience inside Zone C; experience vs
    zone (are new workers spread evenly?); Zone C vs other zones' processing
    times; new vs experienced processing times inside Zone C.
    """
    zone_c = df[df["Zone"] == config.KEY_ZONE]
    rows = [
        chi_square_test(df, "Zone", "Delayed Flag", "Delayed vs Zone"),
        chi_square_test(zone_c, "Worker Experience", "Delayed Flag", "Delayed vs Experience (Zone C)"),
        chi_square_test(df, "Zone", "Worker Experience", "Worker Experience vs Zone"),
    ]
    rows += compare_times(zone_c["Processing Time"], df.loc[df["Zone"] != config.KEY_ZONE, "Processing Time"],
                          "Processing time: Zone C vs other zones")
    rows += compare_times(zone_c.loc[zone_c["New Worker"] == 1, "Processing Time"],
                          zone_c.loc[zone_c["New Worker"] == 0, "Processing Time"],
                          "Processing time: New vs Experienced (Zone C)")
    table = pd.DataFrame(rows)
    table["p_bonferroni"] = bonferroni(table["p_value"])
    table["significant_after_correction"] = table["p_bonferroni"] < config.ALPHA
    return table


def icc_anova(values, clusters) -> float:
    """Intra-cluster correlation (ICC) with the one-way ANOVA estimator, floored at 0.

    ICC says how much more alike two orders from the same cluster are than two
    random orders (0 = no more alike, 1 = identical). It is used to inflate a
    sample size when a trial is randomised by cluster instead of by order.
    """
    frame = pd.DataFrame({"y": np.asarray(values, float), "g": np.asarray(clusters)})
    groups = frame.groupby("g")["y"]
    n_i, means = groups.size(), groups.mean()
    k, n = len(n_i), len(frame)
    if k < 2 or n <= k:
        return 0.0
    msb = float((n_i * (means - frame["y"].mean()) ** 2).sum() / (k - 1))
    msw = float(((frame["y"] - frame["g"].map(means)) ** 2).sum() / (n - k))
    n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return max(0.0, (msb - msw) / denom) if denom > 0 else 0.0
