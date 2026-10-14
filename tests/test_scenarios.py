"""Scenario mechanics: no side effects, reproducible, and logically sensible."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from warehouse import model, scenarios


@pytest.fixture(scope="module")
def effects(real_df):
    return scenarios.effects_from(model.fit_time_models(real_df))


def test_baseline_is_not_modified(real_df, effects):
    before = real_df.copy(deep=True)
    scenarios.run_scenarios(real_df, effects)
    pd.testing.assert_frame_equal(real_df, before)


def test_same_seed_same_results(real_df, effects):
    a = scenarios.run_scenarios(real_df, effects, seed=1)
    b = scenarios.run_scenarios(real_df, effects, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_s1_never_increases_delays(real_df, effects):
    out, _ = scenarios.s1_support_new_c(real_df, effects, 15)
    assert (out["Processing Time"] <= real_df["Processing Time"] + 1e-9).all()
    assert out["Delayed Flag"].sum() <= real_df["Delayed Flag"].sum()


def test_s2_keeps_new_worker_count(real_df, effects):
    out, affected = scenarios.s2_reroster(real_df, effects, 15, np.random.default_rng(0))
    assert (out["Worker Experience"] == "New").sum() == (real_df["Worker Experience"] == "New").sum()
    assert ((out["Zone"] == "C") & (out["Worker Experience"] == "New")).sum() == 0
    assert affected.sum() == 2 * ((real_df["Zone"] == "C") & (real_df["Worker Experience"] == "New")).sum()


def test_s3_with_zero_delta_equals_baseline(real_df, effects):
    base, _ = scenarios.s0_baseline(real_df, effects, 15)
    out, _ = scenarios.s3_layout(real_df, effects, 15, delta=0.0)
    pd.testing.assert_frame_equal(out, base)
    assert (base["Delayed Flag"] == real_df["Delayed Flag"]).all()


def test_power_calculation_matches_statsmodels():
    n = scenarios.sample_size_two_proportions(0.65, 0.45)
    ref = NormalIndPower().solve_power(proportion_effectsize(0.65, 0.45), alpha=0.05, power=0.8)
    assert n == int(np.ceil(ref))


def test_clustered_trial_size_is_at_least_independent(real_df, effects):
    trial = scenarios.trial_design(real_df, effects, 15)
    assert (trial["deff"] >= 1).all()
    assert (trial["orders_per_group_clustered"] >= trial["orders_per_group"]).all()
    assert (trial["clusters_per_arm"] >= 1).all()
