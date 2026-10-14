"""What-if simulations built on effects MEASURED by the models.

Each scenario edits the pick and pack time of the orders it would affect,
using the multipliers estimated in model.py (not invented percentages), and
then recomputes every metric with exactly the same delay definition.

What is deliberately NOT modelled: queues, order arrivals and throughput.
The data has no arrival timestamps or staffing levels, so any claim about
waiting time or orders per hour would be made up. The scenarios therefore
answer "how many orders would cross the delay threshold if each order took
this long?", not "how fast would the warehouse flow?".

The baseline dataframe is never modified: every scenario works on a copy.
"""

from functools import partial

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def effects_from(params) -> dict:
    """Pull the three multipliers the scenarios need out of a fitted-parameter set."""
    return {"m_evening": float(params["m_evening"]), "m_new": float(params["m_new"]),
            "p_new": float(params["p_new"])}


def _finish(df: pd.DataFrame, pick: np.ndarray, pack: np.ndarray, threshold: float) -> pd.DataFrame:
    """Copy the data with new pick/pack times and re-flag delays.

    Processing time changes by exactly the change in pick + pack, so orders a
    scenario does not touch keep their recorded processing time to the digit.
    """
    out = df.copy()
    out["Processing Time"] = (df["Processing Time"].to_numpy()
                              + (pick - df["Pick Time"].to_numpy()) + (pack - df["Pack Time"].to_numpy()))
    out["Pick Time"], out["Pack Time"] = pick, pack
    out["Delayed Flag"] = (out["Processing Time"] > threshold).astype(int)
    return out


def _new_in_zone_c(df: pd.DataFrame) -> np.ndarray:
    """True for orders picked by a New worker in the key zone (Zone C)."""
    return ((df["Zone"] == config.KEY_ZONE) & (df["Worker Experience"] == "New")).to_numpy()


def s0_baseline(df, effects, threshold, rng=None):
    """S0: the observed data, re-flagged with the chosen threshold."""
    pick, pack = df["Pick Time"].to_numpy(float), df["Pack Time"].to_numpy(float)
    return _finish(df, pick, pack, threshold), np.zeros(len(df), bool)


def s1_support_new_c(df, effects, threshold, rng=None):
    """S1: training or pairing makes New workers in Zone C work at experienced speed.

    Only those orders change: pick / m_new and pack / p_new.
    """
    mask = _new_in_zone_c(df)
    pick = np.where(mask, df["Pick Time"] / effects["m_new"], df["Pick Time"])
    pack = np.where(mask, df["Pack Time"] / effects["p_new"], df["Pack Time"])
    return _finish(df, pick, pack, threshold), mask


def s2_reroster(df, effects, threshold, rng):
    """S2: capacity-neutral swap of New workers out of Zone C.

    The New-worker Zone C orders get experienced speed; an equal number of
    Experienced-worker orders in other zones (chosen at random) get new-worker
    speed, and their labels swap, so total staffing is unchanged.
    """
    to_c = _new_in_zone_c(df)
    pool = np.flatnonzero(((df["Zone"] != config.KEY_ZONE) & (df["Worker Experience"] == "Experienced")).to_numpy())
    to_new = np.zeros(len(df), bool)
    to_new[rng.choice(pool, size=int(to_c.sum()), replace=False)] = True
    factor_pick = np.where(to_c, 1 / effects["m_new"], np.where(to_new, effects["m_new"], 1.0))
    factor_pack = np.where(to_c, 1 / effects["p_new"], np.where(to_new, effects["p_new"], 1.0))
    out = _finish(df, df["Pick Time"].to_numpy() * factor_pick,
                  df["Pack Time"].to_numpy() * factor_pack, threshold)
    out.loc[to_c, "Worker Experience"], out.loc[to_new, "Worker Experience"] = "Experienced", "New"
    out["New Worker"] = (out["Worker Experience"] == "New").astype(int)
    return out, to_c | to_new


def s3_layout(df, effects, threshold, rng=None, delta=1.0):
    """S3: better layout/slotting cuts Zone C's base pick time by `delta` minutes.

    Applied through the model, so the saving is larger for orders that also
    carry the evening and new-worker multipliers:
        pick_new = pick - delta * m_evening^[Evening] * m_new^[New]
    A pick never drops below config.MIN_PICK_TIME (and never goes up).
    """
    mask = (df["Zone"] == config.KEY_ZONE).to_numpy()
    evening = (df["Shift"] == "Evening").to_numpy(float)
    new = (df["Worker Experience"] == "New").to_numpy(float)
    cut = delta * effects["m_evening"] ** evening * effects["m_new"] ** new
    pick = df["Pick Time"].to_numpy(float)
    floor = np.minimum(pick, config.MIN_PICK_TIME)
    new_pick = np.where(mask, np.maximum(pick - cut, floor), pick)
    return _finish(df, new_pick, df["Pack Time"].to_numpy(float), threshold), mask & (delta > 0)


def s4_combined(df, effects, threshold, rng=None, delta=config.S4_DELTA):
    """S4: S1 (support new workers in Zone C) followed by S3 with `delta`."""
    step1, mask1 = s1_support_new_c(df, effects, threshold)
    step2, mask2 = s3_layout(step1, effects, threshold, delta=delta)
    return step2, mask1 | mask2


def scenario_specs(deltas=config.S3_DELTAS, s4_delta=config.S4_DELTA) -> list[tuple]:
    """The list of scenarios to run: (code, description, function)."""
    specs = [("S0", "Baseline (observed)", s0_baseline),
             ("S1", "Support New workers in Zone C", s1_support_new_c),
             ("S2", "Re-roster: swap New workers out of Zone C", s2_reroster)]
    specs += [(f"S3 ({d:g} min)", f"Zone C layout: base pick -{d:g} min", partial(s3_layout, delta=d))
              for d in deltas]
    specs.append((f"S4 (S1 + S3 {s4_delta:g} min)", "Combined S1 + layout", partial(s4_combined, delta=s4_delta)))
    return specs


def scenario_row(base: pd.DataFrame, scen: pd.DataFrame, affected: np.ndarray) -> dict:
    """All reported metrics for one scenario compared with the baseline."""
    in_c = scen["Zone"] == config.KEY_ZONE
    avoided = int(base["Delayed Flag"].sum() - scen["Delayed Flag"].sum())
    n_affected = int(affected.sum())
    return {
        "delayed_orders": int(scen["Delayed Flag"].sum()),
        "delay_rate": float(scen["Delayed Flag"].mean()),
        "zone_c_delay_rate": float(scen.loc[in_c, "Delayed Flag"].mean()),
        "delayed_avoided": avoided,
        "mean_processing": float(scen["Processing Time"].mean()),
        "p90_processing": float(scen["Processing Time"].quantile(0.9)),
        "pick_minutes_change": float(scen["Pick Time"].sum() - base["Pick Time"].sum()),
        "orders_affected": n_affected,
        "avoided_per_100_affected": 100 * avoided / n_affected if n_affected else np.nan,
    }


def run_scenarios(df: pd.DataFrame, effects: dict, threshold: float = config.DELAY_THRESHOLD,
                  seed: int = config.RANDOM_SEED, specs=None) -> pd.DataFrame:
    """Run every scenario once with the point-estimate effects.

    Each scenario gets its own random generator seeded with `seed`, so results
    do not depend on the order in which scenarios run.
    """
    specs = specs or scenario_specs()
    base, _ = s0_baseline(df, effects, threshold)
    rows = []
    for code, name, func in specs:
        scen, affected = func(df, effects, threshold, rng=np.random.default_rng(seed))
        rows.append({"scenario": code, "description": name, **scenario_row(base, scen, affected)})
    return pd.DataFrame(rows)


def scenario_uncertainty(df: pd.DataFrame, draws: pd.DataFrame, threshold: float = config.DELAY_THRESHOLD,
                         seed: int = config.RANDOM_SEED, level: float = config.CI_LEVEL) -> pd.DataFrame:
    """Re-run every scenario with each bootstrap parameter set.

    The spread of 'delayed orders avoided' across the runs is a 95% range that
    reflects how uncertain the measured effects are (and, for S2, which orders
    happen to be swapped).
    """
    base_delayed = int((df["Processing Time"] > threshold).sum())
    results = {code: [] for code, _, _ in scenario_specs()}
    for i, params in enumerate(draws.to_dict("records")):
        effects = effects_from(params)
        for code, _, func in scenario_specs():
            scen, _ = func(df, effects, threshold, rng=np.random.default_rng([seed, i]))
            results[code].append(base_delayed - int(scen["Delayed Flag"].sum()))
    tail = (1 - level) / 2
    return pd.DataFrame([{"scenario": code, "avoided_low": float(np.quantile(v, tail)),
                          "avoided_high": float(np.quantile(v, 1 - tail))} for code, v in results.items()])


def rank_scenarios(table: pd.DataFrame) -> pd.DataFrame:
    """Rank scenarios by delayed orders avoided per 100 orders whose process changes.

    'Orders affected' is the disruption measure: an intervention that changes
    how fewer orders are handled is easier to run and to reverse. The baseline
    is excluded because it changes nothing.
    """
    ranked = table[table["scenario"] != "S0"].copy()
    ranked = ranked.sort_values(["avoided_per_100_affected", "delayed_avoided"], ascending=False)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked.reset_index(drop=True)


def threshold_sensitivity(df: pd.DataFrame, effects: dict, thresholds=config.SENSITIVITY_THRESHOLDS,
                          seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Repeat the scenario ranking for several delay thresholds (e.g. 12, 15, 18 min).

    If the best scenario only wins at one arbitrary threshold, the
    recommendation is fragile; if it wins at all of them, it is robust.
    """
    frames = []
    for t in thresholds:
        ranked = rank_scenarios(run_scenarios(df, effects, threshold=t, seed=seed))
        ranked.insert(0, "threshold", t)
        frames.append(ranked[["threshold", "scenario", "delayed_avoided", "avoided_per_100_affected", "rank"]])
    return pd.concat(frames, ignore_index=True)


def sample_size_two_proportions(p1: float, p2: float, alpha: float = config.ALPHA,
                                power: float = config.POWER) -> int:
    """Orders needed PER GROUP to detect a change from rate p1 to p2.

    Two-sided two-proportion test using Cohen's h (arcsine transform), equal
    group sizes. Each group adds variance 1/n, hence the factor 2:
        n = 2 * ((z_(1-alpha/2) + z_power) / h)^2
    """
    h = abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z / h) ** 2))


def s1_sanity_check(df: pd.DataFrame, effects: dict, threshold: float = config.DELAY_THRESHOLD) -> dict:
    """Compare S1's simulated New-worker Zone C delay rate with reality.

    S1 divides each recorded time by the multiplier, which also shrinks that
    order's random noise. If the simulated rate comes out below the rate that
    experienced workers in Zone C actually achieve, the simulation is too
    optimistic and the observed experienced rate is the more realistic target.
    """
    new_c = _new_in_zone_c(df)
    exp_c = ((df["Zone"] == config.KEY_ZONE) & (df["Worker Experience"] == "Experienced")).to_numpy()
    scen, _ = s1_support_new_c(df, effects, threshold)
    return {"observed_new_c_rate": float((df.loc[new_c, "Processing Time"] > threshold).mean()),
            "simulated_new_c_rate": float(scen.loc[new_c, "Delayed Flag"].mean()),
            "observed_experienced_c_rate": float((df.loc[exp_c, "Processing Time"] > threshold).mean())}


def cluster_design(df: pd.DataFrame, threshold: float = config.DELAY_THRESHOLD) -> dict:
    """Preliminary clustering inputs for a trial randomised by worker-shift.

    Worker IDs are not recorded, so each date x shift is used as the cluster.
    Returns the ICC of the delayed flag for New-worker Zone C orders, the
    average cluster size m, the design effect DEFF = 1 + (m - 1) x ICC and the
    number of clusters per day. The real ICC should be re-estimated from pilot data.
    """
    from .stats_tests import icc_anova  # local import keeps scenarios.py standalone
    orders = df[_new_in_zone_c(df)]
    clusters = orders["Date"].astype(str) + " " + orders["Shift"].astype(str)
    icc = icc_anova(orders["Processing Time"] > threshold, clusters)
    m = float(clusters.value_counts().mean())
    return {"icc": icc, "avg_cluster_size": m, "deff": 1 + (m - 1) * icc,
            "clusters_per_day": clusters.nunique() / df["Date"].nunique()}


def trial_design(df: pd.DataFrame, effects: dict, threshold: float = config.DELAY_THRESHOLD) -> pd.DataFrame:
    """Size a trial of S1 on New-worker Zone C orders (preliminary estimate).

    Metric: delay rate of New-worker Zone C orders. Baseline: their observed
    rate. Expected with support: the OBSERVED rate of Experienced workers in
    Zone C (not the S1 simulation, which is too optimistic; see
    s1_sanity_check). A conservative case assumes only half that improvement.
    The independent-orders sample size is inflated by the design effect
    because the trial would be randomised by worker-shift (date x shift here).
    Days assume two equal arms (support on / off) sharing the clusters.
    """
    check, design = s1_sanity_check(df, effects, threshold), cluster_design(df, threshold)
    p1, p_full = check["observed_new_c_rate"], check["observed_experienced_c_rate"]
    per_day = _new_in_zone_c(df).sum() / df["Date"].nunique()
    rows = []
    for case, p2 in (("Expected improvement (to observed Experienced Zone C rate)", p_full),
                     ("Half the expected improvement", (p1 + p_full) / 2)):
        n = sample_size_two_proportions(p1, p2)
        n_clustered = int(np.ceil(n * design["deff"]))
        clusters = int(np.ceil(n_clustered / design["avg_cluster_size"]))
        rows.append({"case": case, "baseline_rate": p1, "target_rate": p2, "orders_per_group": n,
                     "orders_total": 2 * n, "new_zone_c_orders_per_day": float(per_day),
                     "days_needed": int(np.ceil(2 * n / per_day)), **design,
                     "orders_per_group_clustered": n_clustered, "clusters_per_arm": clusters,
                     "days_needed_clustered": int(np.ceil(2 * clusters / design["clusters_per_day"]))})
    return pd.DataFrame(rows)
