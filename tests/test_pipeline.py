"""The saved tables, the report and the README agree with the metric functions,
and the one-command run finishes quickly."""

import subprocess
import sys
import time

import pandas as pd
import pytest

from warehouse import config, metrics, report


def test_saved_tables_match_metric_functions(pipeline, real_df):
    out = pipeline["report_path"].parent / "tables"
    saved = pd.read_csv(out / "by_zone.csv")
    fresh = metrics.summarise_by(real_df, "Zone")
    pd.testing.assert_frame_equal(saved[fresh.columns], fresh, check_dtype=False)
    kpis = pd.read_csv(out / "kpis.csv").set_index("kpi")["value"]
    assert kpis["delay_rate"] == pytest.approx(metrics.kpis(real_df)["delay_rate"])


def test_chart_tables_are_the_metric_tables(pipeline, real_df):
    ze = pipeline["zone_experience"]
    fresh = metrics.summarise_by(real_df, ["Zone", "Worker Experience"])
    pd.testing.assert_series_equal(ze["delay_rate"], fresh["delay_rate"])
    assert len(pipeline["chart_paths"]) == 9 and all(p.exists() for p in pipeline["chart_paths"])


def test_report_quotes_computed_numbers(pipeline, real_df):
    text = pipeline["report_path"].read_text(encoding="utf-8")
    zone = metrics.summarise_by(real_df, "Zone").set_index("Zone")
    assert report.pct(zone.loc["C", "delay_rate"]) in text
    assert f"{metrics.kpis(real_df)['orders']:,} orders" in text


def test_readme_numbers_match(pipeline, real_df):
    readme = config.README.read_text(encoding="utf-8")
    zone = metrics.summarise_by(real_df, "Zone").set_index("Zone")
    ze = metrics.summarise_by(real_df, ["Zone", "Worker Experience"]).set_index(["Zone", "Worker Experience"])
    assert f"**{report.pct(zone.loc['C', 'share_of_delays'], 0)} of all delays**" in readme
    assert report.pct(ze.loc[("C", "New"), "share_of_delays"]) in readme


def test_one_command_runs_under_60_seconds(tmp_path):
    start = time.perf_counter()
    result = subprocess.run([sys.executable, "run_analysis.py", "--no-publish", "--outdir", str(tmp_path)],
                            cwd=config.ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    assert result.returncode == 0, result.stderr
    assert elapsed < 60
    assert (tmp_path / "report.md").exists() and len(list((tmp_path / "charts").glob("*.png"))) == 9


def test_bad_data_stops_with_clear_error(tmp_path):
    bad = pd.read_csv(config.RAW_DATA, encoding="utf-8-sig").head(50)
    bad.loc[0, "Pick Time"] = -2
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    result = subprocess.run([sys.executable, "run_analysis.py", "--data", str(path), "--no-publish",
                             "--outdir", str(tmp_path / "out")], cwd=config.ROOT, capture_output=True, text=True)
    assert result.returncode == 1 and "Pick or Pack time <= 0" in result.stderr


def test_s1_sanity_check_is_reported(pipeline, real_df):
    zone_c = real_df[real_df["Zone"] == "C"]
    experienced_c = zone_c.loc[zone_c["Worker Experience"] == "Experienced", "Delayed Flag"].mean()
    check = pipeline["s1_check"]
    assert check["observed_experienced_c_rate"] == pytest.approx(experienced_c)
    text = pipeline["report_path"].read_text(encoding="utf-8")
    assert "Sanity check (S1)" in text
    assert report.pct(check["simulated_new_c_rate"]) in text and report.pct(experienced_c) in text
    trial = pipeline["trial"].iloc[0]
    assert trial["target_rate"] == pytest.approx(experienced_c)  # trial uses the observed rate, not S1
