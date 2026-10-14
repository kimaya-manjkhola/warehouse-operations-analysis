"""Run the whole analysis with one command.

    python run_analysis.py [--data data/raw_data.csv] [--threshold 15] [--seed 42]

Writes data/clean_data.csv, outputs/tables/*.csv, outputs/charts/*.png and
outputs/report.md, refreshes the generated blocks in README.md and the chart
copies in charts/, and prints a short summary.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from warehouse import charts, config, data, metrics, model, report, scenarios, stats_tests


def step_validate(path: Path, threshold: float) -> dict:
    """Step 1: load, validate (stop on failure) and clean the data."""
    raw = data.load_data(path)
    validation = data.validate_data(raw)
    data.assert_valid(validation)
    return {"df": data.clean_data(raw, threshold), "validation_data": validation}


def step_describe(df: pd.DataFrame) -> dict:
    """Step 2: KPIs and group summaries (delay rate AND share of all delays)."""
    zone = stats_tests.add_wilson_ci(metrics.summarise_by(df, "Zone"))
    shift = metrics.summarise_by(df, "Shift")
    shift = shift.set_index("Shift").loc[list(config.SHIFTS)].reset_index()
    return {"kpis": metrics.kpis(df), "by_zone": zone, "by_shift": shift,
            "by_experience": metrics.summarise_by(df, "Worker Experience"),
            "by_workload": metrics.summarise_by(df, "Workload Band"),
            "by_zone_shift": metrics.summarise_by(df, ["Zone", "Shift"])}


def step_root_cause(df: pd.DataFrame) -> dict:
    """Step 3: Zone x Experience, spread of new workers across zones, test family."""
    ze = stats_tests.add_wilson_ci(metrics.summarise_by(df, ["Zone", "Worker Experience"]))
    spread = pd.crosstab(df["Zone"], df["Worker Experience"])
    spread = pd.DataFrame({"Zone": spread.index, "new_orders": spread["New"].to_numpy(),
                           "experienced_orders": spread["Experienced"].to_numpy(),
                           "share_new": (spread["New"] / spread.sum(axis=1)).to_numpy()})
    per_shift = (df.groupby(["Date", "Shift"], observed=True)
                   .agg(**{"Orders in Shift": ("Orders in Shift", "first"), "delay_rate": ("Delayed Flag", "mean"),
                           "orders": ("Order Id", "size")}).reset_index())
    return {"zone_experience": ze, "experience_by_zone": spread,
            "tests": stats_tests.run_test_suite(df), "per_shift": per_shift}


def step_models(df: pd.DataFrame, seed: int, iterations: int) -> dict:
    """Step 4: fit, bootstrap and validate the effect-separation models."""
    estimates = model.fit_time_models(df)
    draws = model.bootstrap_time_models(df, estimates, iterations=iterations, seed=seed)
    time_params = model.parameter_table(estimates, draws)
    lpm = model.coef_table(model.fit_error_model(df))
    error_coefs = lpm.merge(model.fit_error_logit(df), on="term", how="left")
    residual = model.coef_table(model.pick_residual_check(df, estimates))
    confounding, corr = model.workload_confounding(df)
    return {"estimates": estimates, "draws": draws, "time_params": time_params,
            "fit_stats": model.time_model_fit_stats(df, estimates), "error_coefs": error_coefs,
            "residual_coefs": residual, "confounding": confounding, "evening_workload_corr": corr,
            "validation": model.validation_table(time_params, lpm, residual),
            "shift_mix": model.shift_mix_check(df), "out_of_sample": model.out_of_sample_check(df, seed)}


def step_scenarios(df: pd.DataFrame, estimates: dict, draws: pd.DataFrame, threshold: float, seed: int) -> dict:
    """Step 5: what-if scenarios, their uncertainty, ranking, sensitivity and trial size."""
    effects = scenarios.effects_from(estimates)
    table = scenarios.run_scenarios(df, effects, threshold, seed)
    table = table.merge(scenarios.scenario_uncertainty(df, draws, threshold, seed), on="scenario")
    return {"scenarios": table, "ranking": scenarios.rank_scenarios(table),
            "sensitivity": scenarios.threshold_sensitivity(df, effects, seed=seed),
            "trial": scenarios.trial_design(df, effects, threshold),
            "s1_check": scenarios.s1_sanity_check(df, effects, threshold)}


TABLES = ["validation_data", "by_zone", "by_shift", "by_experience", "by_workload", "by_zone_shift",
          "zone_experience", "experience_by_zone", "tests", "per_shift", "time_params", "draws",
          "error_coefs", "residual_coefs", "confounding", "validation", "shift_mix", "out_of_sample",
          "scenarios", "ranking", "sensitivity", "trial"]


def save_tables(results: dict, out_dir: Path) -> None:
    """Write every result table to outputs/tables/<name>.csv (plus KPIs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        results[name].to_csv(out_dir / f"{name}.csv", index=False)
    pd.Series(results["kpis"]).rename("value").to_csv(out_dir / "kpis.csv", index_label="kpi")
    pd.Series(results["fit_stats"]).rename("value").to_csv(out_dir / "fit_stats.csv", index_label="statistic")
    pd.Series(results["s1_check"]).rename("value").to_csv(out_dir / "s1_sanity_check.csv", index_label="rate")


def make_charts(r: dict, out_dir: Path) -> list[Path]:
    """Draw all nine charts from the computed tables."""
    charts.apply_style()
    return [
        charts.delay_rate_by_zone(r["by_zone"], out_dir),
        charts.share_orders_vs_delays(r["by_zone"], out_dir),
        charts.zone_experience_heatmap(r["zone_experience"], out_dir),
        charts.processing_time_by_zone(r["df"], out_dir),
        charts.workload_confounding(r["per_shift"], out_dir),
        charts.validation_chart(r["validation"], out_dir),
        charts.scenario_comparison(r["scenarios"], r["ranking"].iloc[0]["scenario"], out_dir),
        charts.s3_sensitivity(r["scenarios"], out_dir),
        charts.threshold_sensitivity(r["sensitivity"], out_dir),
    ]


def run(data_path: Path = config.RAW_DATA, threshold: float = config.DELAY_THRESHOLD,
        seed: int = config.RANDOM_SEED, iterations: int = config.BOOTSTRAP_ITERATIONS,
        out_dir: Path = config.OUTPUT_DIR) -> dict:
    """Run steps 1-5, save tables, charts and the report; return all results."""
    np.random.seed(seed)  # belt and braces: every function also takes its own seeded generator
    r = {"threshold": threshold, "seed": seed, "iterations": iterations}
    r.update(step_validate(Path(data_path), threshold))
    r.update(step_describe(r["df"]))
    r.update(step_root_cause(r["df"]))
    r.update(step_models(r["df"], seed, iterations))
    r.update(step_scenarios(r["df"], r["estimates"], r["draws"], threshold, seed))
    out_dir = Path(out_dir)
    save_tables(r, out_dir / "tables")
    r["chart_paths"] = make_charts(r, out_dir / "charts")
    r["report_path"] = out_dir / "report.md"
    r["report_path"].write_text(report.build_report(r), encoding="utf-8")
    return r


def publish(r: dict) -> None:
    """Copy results to the places the repository shows them: data/clean_data.csv,
    charts/ (the README images) and the generated blocks inside README.md."""
    r["df"].to_csv(config.CLEAN_DATA, index=False)
    config.PUBLISHED_CHARTS_DIR.mkdir(exist_ok=True)
    for path in r["chart_paths"]:
        shutil.copy2(path, config.PUBLISHED_CHARTS_DIR / path.name)
    report.update_readme(config.README, report.sections(r))


def print_summary(r: dict, seconds: float) -> None:
    """Print the headline results to the terminal."""
    h = report.headline(r)
    k = r["kpis"]
    print(f"\nOrders: {k['orders']:,} | delayed (> {r['threshold']:g} min): {k['delayed_orders']:,} "
          f"({report.pct(k['delay_rate'])}) | mean processing {k['mean_processing']:.2f} min")
    print(f"Zone C: {report.pct(h['zone_c']['share_of_orders'])} of orders, "
          f"{report.pct(h['zone_c']['share_of_delays'])} of delays")
    print(f"New workers in Zone C: delay rate {report.pct(h['new_c']['delay_rate'])} "
          f"(experienced in Zone C {report.pct(h['exp_c']['delay_rate'])}), "
          f"{report.pct(h['new_c']['share_of_delays'])} of all delays")
    v = r["validation"]
    built = v[v["check_type"] == "built-in effect"]
    print(f"Validation: {int(built['contains_truth'].sum())}/{len(built)} built-in parameters inside 95% CI")
    print(f"Best scenario: {h['best']['scenario']} ({h['best']['description']}), "
          f"{h['best']['delayed_avoided']:,} delayed orders avoided")
    print(f"Finished in {seconds:.1f} s. Report: {r['report_path']}")


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line options (all optional; defaults come from warehouse/config.py)."""
    p = argparse.ArgumentParser(description="Warehouse operations analysis")
    p.add_argument("--data", type=Path, default=config.RAW_DATA, help="raw orders CSV")
    p.add_argument("--threshold", type=float, default=config.DELAY_THRESHOLD, help="delay threshold (min)")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="random seed")
    p.add_argument("--iterations", type=int, default=config.BOOTSTRAP_ITERATIONS, help="bootstrap iterations")
    p.add_argument("--outdir", type=Path, default=config.OUTPUT_DIR, help="where outputs are written")
    p.add_argument("--no-publish", action="store_true",
                   help="do not touch README.md, charts/ or data/clean_data.csv (used by tests)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Entry point: run everything, publish unless told not to, print a summary."""
    args = parse_args(argv)
    start = time.perf_counter()
    try:
        r = run(args.data, args.threshold, args.seed, args.iterations, args.outdir)
    except data.DataValidationError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    if not args.no_publish:
        publish(r)
    print_summary(r, time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
