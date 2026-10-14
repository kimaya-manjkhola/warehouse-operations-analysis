"""Metrics and statistics match hand calculations and reference values."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.proportion import proportion_confint

from warehouse import metrics, stats_tests


def tiny_df() -> pd.DataFrame:
    """250 orders: zone X has 200 orders (90 delayed), zone Y has 50 (10 delayed)."""
    zone = ["X"] * 200 + ["Y"] * 50
    delayed = [1] * 90 + [0] * 110 + [1] * 10 + [0] * 40
    return pd.DataFrame({"Order Id": range(250), "Zone": zone, "Delayed Flag": delayed,
                         "Processing Time": np.where(np.array(delayed) == 1, 20.0, 10.0),
                         "Error Flag": [0] * 250, "Pick Time": 5.0, "Pack Time": 3.0})


def test_kpis_delay_rate():
    k = metrics.kpis(tiny_df())
    assert k["orders"] == 250 and k["delayed_orders"] == 100
    assert k["delay_rate"] == pytest.approx(0.40)


def test_summarise_by_rates_and_shares():
    t = metrics.summarise_by(tiny_df(), "Zone").set_index("Zone")
    assert t.loc["X", "delay_rate"] == pytest.approx(0.45)        # 90 / 200
    assert t.loc["Y", "delay_rate"] == pytest.approx(0.20)        # 10 / 50
    assert t.loc["X", "share_of_delays"] == pytest.approx(0.90)   # 90 / 100
    assert t.loc["X", "share_of_orders"] == pytest.approx(0.80)   # 200 / 250
    assert t["share_of_delays"].sum() == pytest.approx(1.0)


def test_wilson_reference_values():
    low, high = stats_tests.wilson_ci(50, 100)
    assert (low, high) == pytest.approx((0.4038, 0.5962), abs=1e-4)  # textbook value
    for x, n in [(0, 20), (3, 40), (741, 2528)]:
        ref = proportion_confint(x, n, method="wilson")
        assert stats_tests.wilson_ci(x, n) == pytest.approx(ref, abs=1e-10)


def test_cramers_v_and_bonferroni():
    df = pd.DataFrame({"a": [0, 1] * 50, "b": [0, 1] * 50})
    assert stats_tests.chi_square_test(df, "a", "b", "perfect")["effect_size"] == pytest.approx(1.0)
    assert list(stats_tests.bonferroni([0.01, 0.2, 0.6])) == pytest.approx([0.03, 0.6, 1.0])
