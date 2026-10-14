"""Effect-separation models.

Simple group averages mix several causes together (Zone C is slow AND evening
is busy AND some pickers are new). These models estimate each effect while
holding the others fixed, then the estimates are checked against the known
process that generated the synthetic data.

Models
------
* Pick time (non-linear least squares):
      pick = (base_zone + k * items) * m_evening^[Evening] * m_new^[New]
* Pack time (non-linear least squares):
      pack = (a + b * items) * p_new^[New]
* Error: linear probability model (OLS, HC3 robust errors), with logistic
  regression as a robustness check.
* Workload confounding: delayed ~ shift, ~ workload, ~ both.
"""

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit

from . import config

PICK_PARAMS = ["base_A", "base_B", "base_C", "base_D", "k", "m_evening", "m_new"]
PACK_PARAMS = ["pack_a", "pack_b", "p_new"]
PICK_START = [4.0, 5.0, 6.0, 5.0, 0.2, 1.1, 1.1]  # rough starting guesses only
PACK_START = [3.0, 0.1, 1.1]

# ---------------------------------------------------------------- time models


def pick_design(df: pd.DataFrame) -> np.ndarray:
    """Build the inputs of the pick model: one 0/1 column per zone, items,
    an Evening 0/1 flag and a New-worker 0/1 flag (7 rows x n orders)."""
    zones = [(df["Zone"] == z).to_numpy(float) for z in config.ZONES]
    return np.vstack(zones + [df["Items"].to_numpy(float),
                              (df["Shift"] == "Evening").to_numpy(float),
                              (df["Worker Experience"] == "New").to_numpy(float)])


def pick_model(X, base_A, base_B, base_C, base_D, k, m_evening, m_new):
    """Pick time = (zone base + k per item) x evening multiplier x new-worker multiplier.

    The multipliers are raised to the power of a 0/1 flag, so they only apply
    to evening orders / new-worker orders.
    """
    base = X[0] * base_A + X[1] * base_B + X[2] * base_C + X[3] * base_D
    return (base + k * X[4]) * m_evening ** X[5] * m_new ** X[6]


def pack_design(df: pd.DataFrame) -> np.ndarray:
    """Inputs of the pack model: items and a New-worker 0/1 flag."""
    return np.vstack([df["Items"].to_numpy(float),
                      (df["Worker Experience"] == "New").to_numpy(float)])


def pack_model(X, pack_a, pack_b, p_new):
    """Pack time = (a + b per item) x new-worker multiplier."""
    return (pack_a + pack_b * X[0]) * p_new ** X[1]


def r_squared(y, y_hat) -> float:
    """Share of the variation in y that the model explains (1 = perfect)."""
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))


def _fit(model, X, y, names, start) -> dict:
    """Fit one non-linear model with scipy's curve_fit and return {name: value}."""
    params, _ = curve_fit(model, X, y, p0=start, maxfev=20000)
    return dict(zip(names, map(float, params)))


def fit_time_models(df: pd.DataFrame, start: dict | None = None) -> dict:
    """Fit the pick and pack models and return all 10 parameters plus R-squared.

    `start` lets the bootstrap begin from the full-data estimates, which makes
    each refit faster without changing the answer.
    """
    start = start or {}
    pick = _fit(pick_model, pick_design(df), df["Pick Time"].to_numpy(float), PICK_PARAMS,
                [start.get(p, s) for p, s in zip(PICK_PARAMS, PICK_START)])
    pack = _fit(pack_model, pack_design(df), df["Pack Time"].to_numpy(float), PACK_PARAMS,
                [start.get(p, s) for p, s in zip(PACK_PARAMS, PACK_START)])
    return {**pick, **pack}


def predict_pick(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Predicted pick time for every order from fitted parameters."""
    return pick_model(pick_design(df), *[params[p] for p in PICK_PARAMS])


def predict_pack(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Predicted pack time for every order from fitted parameters."""
    return pack_model(pack_design(df), *[params[p] for p in PACK_PARAMS])


def time_model_fit_stats(df: pd.DataFrame, params: dict) -> dict:
    """R-squared and mean absolute error of both time models on `df`."""
    pick_hat, pack_hat = predict_pick(df, params), predict_pack(df, params)
    return {"pick_r2": r_squared(df["Pick Time"], pick_hat),
            "pick_mae": float(np.mean(np.abs(df["Pick Time"] - pick_hat))),
            "pack_r2": r_squared(df["Pack Time"], pack_hat),
            "pack_mae": float(np.mean(np.abs(df["Pack Time"] - pack_hat)))}


def bootstrap_time_models(df: pd.DataFrame, estimates: dict,
                          iterations: int = config.BOOTSTRAP_ITERATIONS,
                          seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Refit both time models on `iterations` resamples of the orders.

    Each resample draws orders with replacement, so the spread of the refitted
    parameters shows how much they could move by chance. One row per resample.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(iterations):
        sample = df.iloc[rng.integers(0, len(df), len(df))]
        rows.append(fit_time_models(sample, start=estimates))
    return pd.DataFrame(rows)


def parameter_table(estimates: dict, draws: pd.DataFrame,
                    level: float = config.CI_LEVEL) -> pd.DataFrame:
    """Estimate and percentile-bootstrap confidence interval for each parameter."""
    tail = (1 - level) / 2
    return pd.DataFrame({
        "parameter": list(draws.columns),
        "estimate": [estimates[p] for p in draws.columns],
        "ci_low": draws.quantile(tail).to_numpy(),
        "ci_high": draws.quantile(1 - tail).to_numpy(),
    })


def out_of_sample_check(df: pd.DataFrame, seed: int = config.RANDOM_SEED,
                        train_share: float = config.TRAIN_SHARE) -> pd.DataFrame:
    """Fit the pick model on a random 80% of orders and score it on the other 20%.

    A model that only memorised the data would score much worse on unseen
    orders. A naive 'average pick time of the zone' predictor is shown for
    comparison so the R-squared has a reference point.
    """
    rng = np.random.default_rng(seed)
    is_train = rng.random(len(df)) < train_share
    train, test = df[is_train], df[~is_train]
    params = fit_time_models(train)
    zone_mean = train.groupby("Zone")["Pick Time"].mean()
    rows = []
    for name, part in (("train (80%)", train), ("test (20%)", test)):
        naive = part["Zone"].map(zone_mean)
        rows.append({"sample": name, "orders": len(part),
                     "model_mae": float(np.mean(np.abs(part["Pick Time"] - predict_pick(part, params)))),
                     "model_r2": r_squared(part["Pick Time"], predict_pick(part, params)),
                     "zone_average_mae": float(np.mean(np.abs(part["Pick Time"] - naive))),
                     "zone_average_r2": r_squared(part["Pick Time"], naive)})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- regressions


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Short snake_case column names for regression formulas.

    Orders in Shift is centred on its mean and expressed per 10 orders, so its
    coefficient reads 'effect of 10 more orders in the shift' and the
    intercept stays a sensible baseline rate.
    """
    workload = df["Orders in Shift"].astype(float)
    return pd.DataFrame({
        "zone": df["Zone"], "shift": df["Shift"], "experience": df["Worker Experience"],
        "category": df["Product Category"], "priority": df["Priority"],
        "workload_10": (workload - workload.mean()) / 10,
        "error": df["Error Flag"].astype(float), "delayed": df["Delayed Flag"].astype(float),
        "evening": (df["Shift"] == "Evening").astype(float),
    })


RHS_TERMS = ("C(zone, Treatment('A')) + C(shift, Treatment('Morning')) + "
             "C(experience, Treatment('Experienced')) + C(category) + "
             "C(priority, Treatment('Standard')) + workload_10")

_TERM = re.compile(r"C\((\w+)[^\]]*\)\[T\.(.+)\]")


def term_key(term: str) -> str:
    """Turn a statsmodels term like "C(zone, Treatment('A'))[T.C]" into "zone_C"."""
    match = _TERM.fullmatch(term)
    return f"{match.group(1)}_{match.group(2)}" if match else term


def coef_table(result, scale: float = 100.0) -> pd.DataFrame:
    """Coefficients, 95% CIs and p-values of a fitted regression, multiplied by
    `scale` (100 turns a probability change into percentage points)."""
    ci = result.conf_int(alpha=1 - config.CI_LEVEL)
    return pd.DataFrame({
        "term": [term_key(t) for t in result.params.index],
        "estimate": result.params.to_numpy() * scale,
        "ci_low": ci.iloc[:, 0].to_numpy() * scale,
        "ci_high": ci.iloc[:, 1].to_numpy() * scale,
        "p_value": result.pvalues.to_numpy(),
    })


def fit_error_model(df: pd.DataFrame):
    """Linear probability model: Error Flag on every recorded factor.

    OLS on a 0/1 outcome gives coefficients that are directly 'change in error
    probability'. HC3 robust standard errors fix the unequal variance that a
    0/1 outcome always has.
    """
    return smf.ols(f"error ~ {RHS_TERMS}", data=model_frame(df)).fit(cov_type="HC3")


def fit_error_logit(df: pd.DataFrame) -> pd.DataFrame:
    """Logistic regression robustness check, reported as average marginal effects
    in percentage points so it can be compared with the linear model."""
    result = smf.logit(f"error ~ {RHS_TERMS}", data=model_frame(df)).fit(disp=False)
    frame = result.get_margeff(at="overall").summary_frame(alpha=1 - config.CI_LEVEL)
    return pd.DataFrame({
        "term": [term_key(t) for t in frame.index],
        "logit_ame": frame.iloc[:, 0].to_numpy() * 100,
        "logit_ci_low": frame.iloc[:, 4].to_numpy() * 100,
        "logit_ci_high": frame.iloc[:, 5].to_numpy() * 100,
        "logit_p_value": frame.iloc[:, 3].to_numpy(),
    })


def pick_residual_check(df: pd.DataFrame, params: dict):
    """Do Category, Priority or workload explain what the pick model misses?

    Regresses log(actual / predicted pick time) on them. Coefficients x 100 are
    roughly '% change in pick time'. The true process has no such effects, so
    all should be close to zero.
    """
    frame = model_frame(df)
    frame["log_ratio"] = np.log(df["Pick Time"].to_numpy() / predict_pick(df, params))
    formula = "log_ratio ~ C(category) + C(priority, Treatment('Standard')) + workload_10"
    return smf.ols(formula, data=frame).fit(cov_type="HC3")


def workload_confounding(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Is 'busy shifts cause delays' real, or is it the evening shift in disguise?

    Fits Delayed Flag on (1) shift only, (2) workload only, (3) both, and
    returns the Evening and workload coefficients from each (in percentage
    points), plus the correlation between Evening and Orders in Shift.
    """
    frame = model_frame(df)
    specs = {"Shift only": "C(shift, Treatment('Morning'))",
             "Workload only": "workload_10",
             "Shift + workload": "C(shift, Treatment('Morning')) + workload_10"}
    rows = []
    for name, rhs in specs.items():
        table = coef_table(smf.ols(f"delayed ~ {rhs}", data=frame).fit(cov_type="HC3"))
        for term in ("shift_Evening", "workload_10"):
            hit = table[table["term"] == term]
            if not hit.empty:
                rows.append({"model": name, **hit.iloc[0].to_dict()})
    corr = float(np.corrcoef(frame["evening"], df["Orders in Shift"])[0, 1])
    return pd.DataFrame(rows), corr

# ---------------------------------------------------------------- validation

TERM_LABELS = {
    "Intercept": "Baseline error rate (reference group*)",
    "workload_10": "Orders in Shift (+10 orders)",
    "priority_Priority": "Priority order (vs Standard)",
    "experience_New": "New worker",
}


def label(term: str) -> str:
    """Readable name for a regression term, e.g. 'zone_C' -> 'Zone C'."""
    if term in TERM_LABELS:
        return TERM_LABELS[term]
    variable, _, level = term.partition("_")
    names = {"zone": "Zone", "shift": "Shift", "experience": "Worker",
             "category": "Category", "priority": "Priority"}
    return f"{names.get(variable, variable)} {level}".strip()


def true_error_pp(term: str) -> float:
    """True effect (percentage points) of a term in the documented error process.

    Terms not mentioned in the process (zones B and D, afternoon, category,
    priority, workload) have a true effect of zero.
    """
    truth = {"Intercept": config.TRUE_ERROR_PP["baseline"],
             f"zone_{config.KEY_ZONE}": config.TRUE_ERROR_PP["zone_C"],
             "shift_Evening": config.TRUE_ERROR_PP["evening"],
             "experience_New": config.TRUE_ERROR_PP["new_worker"]}
    return truth.get(term, 0.0)


PARAM_UNITS = {"base_A": "minutes", "base_B": "minutes", "base_C": "minutes", "base_D": "minutes",
               "k": "minutes per item", "m_evening": "multiplier (x)", "m_new": "multiplier (x)",
               "p_new": "multiplier (x)"}


def _rows(section: str, table: pd.DataFrame, names, truths, units) -> pd.DataFrame:
    """Shape one block of the validation table (one row per parameter)."""
    out = pd.DataFrame({"section": section, "parameter": list(names), "units": units,
                        "true_value": list(truths), "estimate": table["estimate"].to_numpy(),
                        "ci_low": table["ci_low"].to_numpy(), "ci_high": table["ci_high"].to_numpy(),
                        "p_value": table["p_value"].to_numpy() if "p_value" in table else np.nan})
    out["check_type"] = np.where(out["true_value"] == 0, "no effect", "built-in effect")
    return out


def validation_table(time_params: pd.DataFrame, error_coefs: pd.DataFrame,
                     residual_coefs: pd.DataFrame) -> pd.DataFrame:
    """The centrepiece: every estimate next to the value used to generate the data.

    `contains_truth` says whether the 95% CI covers the true value.
    `excludes_zero` flags 'significant' results; for a 'no effect' row that
    would be a false positive, which happens about 1 time in 20 by chance.
    """
    pick = time_params.set_index("parameter").loc[PICK_PARAMS].reset_index()
    pack = time_params.set_index("parameter").loc[["p_new"]].reset_index()
    resid = residual_coefs[residual_coefs["term"] != "Intercept"]
    blocks = [
        _rows("Pick time model", pick, PICK_PARAMS, [config.TRUE_PICK[p] for p in PICK_PARAMS],
              [PARAM_UNITS[p] for p in PICK_PARAMS]),
        _rows("Pack time model", pack, ["p_new"], [config.TRUE_PACK["p_new"]], PARAM_UNITS["p_new"]),
        _rows("Error model (LPM)", error_coefs, [label(t) for t in error_coefs["term"]],
              [true_error_pp(t) for t in error_coefs["term"]], "percentage points"),
        _rows("Pick time: no-effect checks", resid, [label(t) for t in resid["term"]],
              [0.0] * len(resid), "% of pick time"),
    ]
    table = pd.concat(blocks, ignore_index=True)
    table["contains_truth"] = (table["ci_low"] <= table["true_value"]) & (table["true_value"] <= table["ci_high"])
    table["excludes_zero"] = (table["ci_low"] > 0) | (table["ci_high"] < 0)
    return table


def shift_mix_check(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the observed shift mix with the documented 30/30/40 split."""
    from .stats_tests import wilson_ci  # local import keeps model.py standalone
    counts = df["Shift"].value_counts().reindex(config.SHIFTS)
    low, high = wilson_ci(counts.to_numpy(), len(df))
    table = pd.DataFrame({"shift": counts.index, "orders": counts.to_numpy(),
                          "share": counts.to_numpy() / len(df), "ci_low": low, "ci_high": high,
                          "true_share": [config.TRUE_SHIFT_MIX[s] for s in config.SHIFTS]})
    table["contains_truth"] = (table["ci_low"] <= table["true_share"]) & (table["true_share"] <= table["ci_high"])
    return table
