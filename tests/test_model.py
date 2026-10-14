"""Parameter recovery: the models must get back parameters we planted ourselves."""

import pytest

from tests.simulate import TRUE
from warehouse import model


def test_time_model_recovers_true_parameters(simulated_df):
    est = model.fit_time_models(simulated_df)
    for zone, value in TRUE["base"].items():
        assert est[f"base_{zone}"] == pytest.approx(value, abs=0.15)
    assert est["k"] == pytest.approx(TRUE["k"], abs=0.02)
    assert est["m_evening"] == pytest.approx(TRUE["m_evening"], abs=0.02)
    assert est["m_new"] == pytest.approx(TRUE["m_new"], abs=0.02)
    assert est["p_new"] == pytest.approx(TRUE["p_new"], abs=0.02)
    assert est["pack_a"] == pytest.approx(TRUE["pack_a"], abs=0.15)
    assert est["pack_b"] == pytest.approx(TRUE["pack_b"], abs=0.03)


def test_error_model_recovers_effects(simulated_df):
    coefs = model.coef_table(model.fit_error_model(simulated_df)).set_index("term")["estimate"]
    assert coefs["zone_C"] == pytest.approx(TRUE["error"]["zone_C"] * 100, abs=1.5)
    assert coefs["experience_New"] == pytest.approx(TRUE["error"]["new"] * 100, abs=1.5)
    assert coefs["shift_Evening"] == pytest.approx(TRUE["error"]["evening"] * 100, abs=2.0)
    for no_effect in ("zone_B", "zone_D", "priority_Priority", "workload_10"):
        assert abs(coefs[no_effect]) < 1.5


def test_workload_effect_vanishes_once_shift_is_controlled(real_df):
    table, corr = model.workload_confounding(real_df)
    alone = table[(table["model"] == "Workload only") & (table["term"] == "workload_10")].iloc[0]
    both = table[(table["model"] == "Shift + workload") & (table["term"] == "workload_10")].iloc[0]
    assert alone["p_value"] < 0.001 and both["p_value"] > 0.05
    assert abs(both["estimate"]) < abs(alone["estimate"]) / 3
    assert corr > 0.5


def test_out_of_sample_close_to_in_sample(real_df):
    oos = model.out_of_sample_check(real_df).set_index("sample")
    assert oos.loc["test (20%)", "model_r2"] > 0.7
    assert abs(oos.loc["test (20%)", "model_r2"] - oos.loc["train (80%)", "model_r2"]) < 0.05
