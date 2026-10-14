"""Write the Markdown report (and the README's result blocks) from computed results.

No number in the report is typed by hand: every figure is read from the
`results` dictionary produced by run_analysis.py. The README contains blocks
marked <!-- BEGIN:name --> ... <!-- END:name --> that are regenerated on every
run, so the GitHub front page can never drift from the code's output.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------- formatting


def pct(x: float, dp: int = 1) -> str:
    """0.3378 -> '33.8%'."""
    return f"{x * 100:.{dp}f}%"


def num(x: float, dp: int = 2) -> str:
    """Format a number with thousands separators and `dp` decimals."""
    return f"{x:,.{dp}f}"


def md_table(df: pd.DataFrame) -> str:
    """Render a dataframe as a GitHub Markdown table (no extra dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def with_unit(value: float, unit: str) -> str:
    """Signed value with its unit: '+1.22 percentage points', '-0.82% of pick time'."""
    return f"{value:+.2f}{unit}" if unit.startswith("%") else f"{value:+.2f} {unit}"


def ci(low: float, high: float, fmt=pct) -> str:
    """'[low, high]' with the same formatter as the estimate."""
    return f"[{fmt(low)}, {fmt(high)}]"


def _row(table: pd.DataFrame, col: str, value) -> pd.Series:
    """The single row of `table` where `col` equals `value`."""
    return table[table[col] == value].iloc[0]

# ---------------------------------------------------------------- headline facts


def headline(r: dict) -> dict:
    """The handful of facts quoted in several places, computed once."""
    zone_c = _row(r["by_zone"], "Zone", config.KEY_ZONE)
    ze = r["zone_experience"]
    new_c = ze[(ze["Zone"] == config.KEY_ZONE) & (ze["Worker Experience"] == "New")].iloc[0]
    exp_c = ze[(ze["Zone"] == config.KEY_ZONE) & (ze["Worker Experience"] == "Experienced")].iloc[0]
    ranking = r["ranking"]
    best = ranking.iloc[0]
    biggest = ranking.sort_values("delayed_avoided", ascending=False).iloc[0]
    out_test = _row(r["out_of_sample"], "sample", "test (20%)")
    return {"zone_c": zone_c, "new_c": new_c, "exp_c": exp_c, "best": best, "biggest": biggest,
            "out_test": out_test, "kpis": r["kpis"]}


def unc(r: dict, scenario: str) -> str:
    """Bootstrap 95% range of delayed orders avoided, e.g. '356-369'."""
    u = _row(r["scenarios"], "scenario", scenario)
    return f"{u['avoided_low']:,.0f}-{u['avoided_high']:,.0f}"

# ---------------------------------------------------------------- sections


def section_key_results(r: dict) -> str:
    """Short bullet list of the main computed numbers (README 'Key results' box)."""
    h, k = headline(r), r["kpis"]
    v = r["validation"]
    built = v[v["check_type"] == "built-in effect"]
    c = r["confounding"]
    conf = c[(c["model"] == "Shift + workload") & (c["term"] == "workload_10")].iloc[0]
    return "\n".join([
        f"> - **{k['orders']:,} orders**, **{pct(k['delay_rate'])} delayed** (>{r['threshold']:g} min), "
        f"mean processing time **{num(k['mean_processing'], 1)} min**.",
        f"> - **Zone C** handles {pct(h['zone_c']['share_of_orders'], 0)} of orders but "
        f"**{pct(h['zone_c']['share_of_delays'], 0)} of all delays**.",
        f"> - **Root cause:** New workers in Zone C are delayed on **{pct(h['new_c']['delay_rate'], 0)}** of "
        f"orders vs {pct(h['exp_c']['delay_rate'], 0)} for experienced workers there. They handle "
        f"{pct(h['new_c']['share_of_orders'])} of orders but cause **{pct(h['new_c']['share_of_delays'])} of all delays**.",
        f"> - **Workload is a confounder:** once shift is controlled for, 10 extra orders in a shift change the "
        f"delay rate by {num(conf['estimate'], 2)} pp (95% CI {num(conf['ci_low'], 2)} to {num(conf['ci_high'], 2)}, "
        f"p = {conf['p_value']:.2f}).",
        f"> - **Validated model:** {int(built['contains_truth'].sum())} of {len(built)} built-in parameters have a 95% CI "
        f"containing the true value; pick-time model R² = {num(r['fit_stats']['pick_r2'])} "
        f"(out-of-sample {num(h['out_test']['model_r2'])}).",
        f"> - **Best first step:** {h['best']['description']} ({h['best']['scenario']}) avoids "
        f"**{h['best']['delayed_avoided']:,.0f} delayed orders a month** (95% range {unc(r, h['best']['scenario'])}) "
        f"by changing only {h['best']['orders_affected']:,} orders.",
    ])


def section_describe(r: dict) -> str:
    """Step 2 tables: zone, shift, experience and workload band summaries."""
    def fmt(t: pd.DataFrame, key) -> pd.DataFrame:
        """Turn a summarise_by table into readable columns (percentages, minutes)."""
        keys = [key] if isinstance(key, str) else list(key)
        return pd.DataFrame({
            " x ".join(keys): t[keys].astype(str).agg(" / ".join, axis=1),
            "Orders": t["orders"].map("{:,}".format), "Share of orders": t["share_of_orders"].map(pct),
            "Delay rate": t["delay_rate"].map(pct), "Share of all delays": t["share_of_delays"].map(pct),
            "Mean (min)": t["mean_processing"].map(lambda x: num(x, 1)),
            "P90 (min)": t["p90_processing"].map(lambda x: num(x, 1)),
            "Error rate": t["error_rate"].map(pct)})
    parts = [("Zone", "by_zone"), ("Shift", "by_shift"), ("Worker Experience", "by_experience"),
             ("Workload Band", "by_workload")]
    blocks = [f"**By {key.lower()}**\n\n" + md_table(fmt(r[name], key)) for key, name in parts]
    blocks.append("**Zone x Shift**\n\n" + md_table(fmt(r["by_zone_shift"], ("Zone", "Shift"))))
    return "\n\n".join(blocks)


def section_root_cause(r: dict) -> str:
    """Step 3: Zone x Experience table, evenness of new workers, and the test family."""
    h, ze = headline(r), r["zone_experience"]
    table = pd.DataFrame({
        "Zone": ze["Zone"], "Worker": ze["Worker Experience"], "Orders": ze["orders"].map("{:,}".format),
        "Delay rate": ze["delay_rate"].map(pct),
        "95% CI": [ci(a, b) for a, b in zip(ze["delay_rate_ci_low"], ze["delay_rate_ci_high"])],
        "Share of all delays": ze["share_of_delays"].map(pct)})
    spread = r["experience_by_zone"]
    even = ", ".join(f"{z}: {pct(s)}" for z, s in zip(spread["Zone"], spread["share_new"]))
    chi = _row(r["tests"], "test", "Worker Experience vs Zone")
    tests = r["tests"].assign(
        p=r["tests"]["p_value"].map(lambda p: f"{p:.2g}"),
        p_adj=r["tests"]["p_bonferroni"].map(lambda p: f"{p:.2g}"),
        effect=[f"{m} = {e:.2f}" for m, e in zip(r["tests"]["effect_measure"], r["tests"]["effect_size"])])
    return "\n\n".join([
        md_table(table),
        f"New workers are spread evenly across zones (share of each zone's orders handled by New workers: {even}; "
        f"chi-square p = {chi['p_value']:.2f}, Cramer's V = {chi['effect_size']:.3f}). So Zone C is not slow because it "
        f"gets more new workers: new workers are slow **everywhere**, and Zone C's slower base pick time pushes their "
        f"orders over the threshold far more often.",
        f"**New workers in Zone C handle {pct(h['new_c']['share_of_orders'])} of all orders but account for "
        f"{pct(h['new_c']['share_of_delays'])} of all delayed orders.**",
        "Statistical tests (Bonferroni-corrected across the family):\n\n"
        + md_table(tests[["test", "method", "p", "p_adj", "effect", "significant_after_correction"]]
                   .rename(columns={"p_adj": "p (Bonferroni)", "significant_after_correction": "significant"})),
    ])


def section_models(r: dict) -> str:
    """Step 4: parameter estimates, error model, confounding check, out-of-sample check."""
    tp = r["time_params"]
    params = pd.DataFrame({"Parameter": tp["parameter"], "Estimate": tp["estimate"].map(lambda x: num(x, 3)),
                           "95% bootstrap CI": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(tp["ci_low"], tp["ci_high"])]})
    conf = r["confounding"]
    conf_t = pd.DataFrame({"Model": conf["model"], "Term": conf["term"].map({"shift_Evening": "Evening (vs Morning)",
                                                                              "workload_10": "+10 orders in shift"}),
                           "Effect on delay rate (pp)": conf["estimate"].map(lambda x: num(x, 2)),
                           "95% CI": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(conf["ci_low"], conf["ci_high"])],
                           "p-value": conf["p_value"].map(lambda p: f"{p:.2g}")})
    oos = r["out_of_sample"]
    oos_t = pd.DataFrame({"Sample": oos["sample"], "Orders": oos["orders"].map("{:,}".format),
                          "Model MAE (min)": oos["model_mae"].map(lambda x: num(x, 3)),
                          "Model R²": oos["model_r2"].map(lambda x: num(x, 3)),
                          "Zone-average MAE (min)": oos["zone_average_mae"].map(lambda x: num(x, 3)),
                          "Zone-average R²": oos["zone_average_r2"].map(lambda x: num(x, 3))})
    fs = r["fit_stats"]
    return "\n\n".join([
        f"**Time models** (pick R² = {num(fs['pick_r2'], 3)}, pack R² = {num(fs['pack_r2'], 3)}; "
        f"{r['iterations']} bootstrap resamples):\n\n" + md_table(params),
        f"**Workload confounding check** (correlation between Evening and Orders in Shift = "
        f"{num(r['evening_workload_corr'], 2)}):\n\n" + md_table(conf_t),
        "**Out-of-sample check of the pick model** (fit on a random 80%, scored on the other 20%):\n\n"
        + md_table(oos_t),
    ])


def section_validation(r: dict) -> str:
    """Step 4e: the validation table plus an honest reading of any misses."""
    v = r["validation"]
    table = pd.DataFrame({
        "Section": v["section"], "Parameter": v["parameter"], "Unit": v["units"],
        "True": v["true_value"].map(lambda x: f"{x:g}"),
        "Estimate": v["estimate"].map(lambda x: num(x, 3)),
        "95% CI": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(v["ci_low"], v["ci_high"])],
        "CI contains truth": v["contains_truth"].map({True: "yes", False: "**no**"})})
    note = ("\\* Reference group for the error model: Zone A, Morning shift, Experienced worker, Clothing, "
            "Standard priority, average workload.")
    return md_table(table) + "\n\n" + note + "\n\n" + validation_commentary(r)


def validation_commentary(r: dict) -> str:
    """Plain-English reading of the validation table, written from its contents."""
    v = r["validation"]
    built, null = v[v["check_type"] == "built-in effect"], v[v["check_type"] == "no effect"]
    missed = built[~built["contains_truth"]]
    false_pos = null[null["excludes_zero"]]
    time_rows = built[built["section"].str.contains("time")]
    err_rows = built[built["section"] == "Error model (LPM)"]
    lines = [f"- {int(built['contains_truth'].sum())} of {len(built)} built-in parameters have a 95% CI containing "
             f"the true value. Every time-model parameter is within "
             f"{pct((time_rows['estimate'] / time_rows['true_value'] - 1).abs().max())} of its true value; error-model "
             f"effects are within {(err_rows['estimate'] - err_rows['true_value']).abs().max():.2f} percentage points "
             "(errors are rare, so these estimates are much noisier)."]
    if len(missed):
        names = ", ".join(f"{m.parameter} ({m.estimate:.3f} vs {m.true_value:g})" for m in missed.itertuples())
        lines.append(f"- Missed: {names}. The estimates are close in practical terms, but the bootstrap CI only "
                     "reflects sampling noise. It cannot capture details of how Excel generated the data (for example "
                     "rounding to 0.1 min or the exact shape of the noise), so a very narrow CI can sit just beside "
                     "the truth.")
    lines.append(f"- {len(null) - len(false_pos)} of {len(null)} 'no effect' checks correctly include zero.")
    if len(false_pos):
        names = ", ".join(f"{m.parameter} in '{m.section}' ({with_unit(m.estimate, m.units)}, p = {m.p_value:.3f})"
                          for m in false_pos.itertuples())
        n_adj = int((false_pos["p_value"] * len(null) < config.ALPHA).sum())
        pick_fp = false_pos[false_pos["section"].str.startswith("Pick")]
        mean_pick = r["kpis"]["mean_pick"]
        minutes = ", ".join(f"{m.parameter}: {m.estimate:+.2f}% of an average {mean_pick:.2f}-minute "
                            f"pick = about {abs(m.estimate) / 100 * mean_pick:.2f} minutes"
                            for m in pick_fp.itertuples())
        lines.append(f"- **False positives:** {names}. With {len(null)} no-effect checks at the 5% level, about "
                     f"{len(null) * config.ALPHA:.1f} would be expected to look 'significant' by chance alone. After a "
                     f"Bonferroni correction across these {len(null)} checks, {n_adj} still "
                     f"{'looks' if n_adj == 1 else 'look'} significant"
                     f"{' (' + minutes + ')' if minutes else ''}. Category "
                     "effects share the same reference group (Clothing), so they tend to fail together. Any surviving "
                     "effect is tiny in practical terms and changes no conclusion; this is exactly why we test many "
                     "things cautiously and why a controlled trial is needed before acting.")
    return "\n".join(lines)


def section_scenarios(r: dict) -> str:
    """Step 5: scenario results table."""
    s = r["scenarios"]
    table = pd.DataFrame({
        "Scenario": s["scenario"], "What changes": s["description"],
        "Delay rate": s["delay_rate"].map(pct), "Zone C delay rate": s["zone_c_delay_rate"].map(pct),
        "Delayed orders avoided / month": s["delayed_avoided"].map("{:,}".format),
        "95% range": [f"{a:,.0f}-{b:,.0f}" for a, b in zip(s["avoided_low"], s["avoided_high"])],
        "Mean / P90 (min)": [f"{a:.2f} / {b:.2f}" for a, b in zip(s["mean_processing"], s["p90_processing"])],
        "Pick minutes change": s["pick_minutes_change"].map(lambda x: f"{x:+,.0f}"),
        "Orders changed": s["orders_affected"].map("{:,}".format),
        "Avoided per 100 changed": s["avoided_per_100_affected"].map(lambda x: "-" if np.isnan(x) else f"{x:.1f}")})
    return md_table(table) + "\n\n" + s1_sanity_sentence(r)


def s1_sanity_sentence(r: dict) -> str:
    """Compare S1's simulated New-worker Zone C delay rate with the observed
    Experienced Zone C rate, and explain any gap."""
    c = r["s1_check"]
    if c["simulated_new_c_rate"] < c["observed_experienced_c_rate"]:
        verdict = ("This is lower than experienced workers actually achieve, which is unrealistic: dividing each "
                   "recorded time by the multiplier also shrinks that order's random noise, so the simulated times "
                   "are less spread out than real experienced times. The trial design therefore uses the observed "
                   "Experienced Zone C rate as the expected result instead.")
    else:
        verdict = "This is not better than experienced workers actually achieve, so the simulation looks plausible."
    return (f"**Sanity check (S1):** simulated delay rate of New-worker Zone C orders = "
            f"{pct(c['simulated_new_c_rate'])}, observed Experienced Zone C delay rate = "
            f"{pct(c['observed_experienced_c_rate'])}. {verdict}")


def section_sensitivity(r: dict) -> str:
    """Threshold sensitivity of the ranking, stated plainly."""
    sens = r["sensitivity"]
    winners = sens[sens["rank"] == 1].set_index("threshold")["scenario"]
    lines = [f"- At {t:g} min the top-ranked scenario is **{s}**." for t, s in winners.items()]
    verdict = ("The best scenario is the same at every threshold, so the ranking is robust."
               if winners.nunique() == 1 else
               "The best scenario changes with the threshold, so the ranking depends on how 'delayed' is defined "
               "and should be confirmed with the business's real service target.")
    negatives = sens[sens["delayed_avoided"] < 0]
    if len(negatives):
        lines.append("- " + "; ".join(f"{n.scenario} at {n.threshold:g} min **adds** {-n.delayed_avoided:,} delays"
                                       for n in negatives.itertuples())
                     + " (moving slower pickers into faster zones pushes their orders over a stricter threshold).")
    return "\n".join(lines + ["", verdict])


def biggest_sentence(big: pd.Series, best: pd.Series) -> str:
    """Explain how the largest-total scenario compares with the most efficient one."""
    if big["scenario"] == best["scenario"]:
        return "It is also the scenario with the largest total reduction."
    layout = " and depends on a physical layout change whose size (delta) is an assumption, not a measurement" \
        if big["scenario"].startswith(("S3", "S4")) else ""
    return (f"The largest total reduction comes from {big['scenario']} ({big['delayed_avoided']:,} avoided), but it "
            f"changes {big['orders_affected']:,} orders{layout}. It is a sensible second phase once the first step "
            "has been tested.")


def sensitivity_sentence(r: dict, scenario: str) -> str:
    """One sentence on whether `scenario` stays top-ranked at every threshold."""
    winners = r["sensitivity"][r["sensitivity"]["rank"] == 1].set_index("threshold")["scenario"]
    wins = [f"{t:g}" for t, s in winners.items() if s == scenario] or ["none of the thresholds"]
    others = [f"{s} at {t:g} min" for t, s in winners.items() if s != scenario]
    if not others:
        return f"{scenario} ranks first at every tested threshold ({', '.join(wins)} min), so this is robust."
    return (f"{scenario} ranks first at {' and '.join(wins)} min, but not everywhere ({'; '.join(others)} ranks first), "
            "so the choice should be confirmed against the warehouse's real service target.")


def section_recommendation(r: dict) -> str:
    """Recommendation and trial design written from the ranking and power analysis."""
    h, trial = headline(r), r["trial"]
    best, big = h["best"], h["biggest"]
    full, half = trial.iloc[0], trial.iloc[1]
    days = max(int(half["days_needed"]), config.TRIAL_MIN_DAYS)
    lines = [
        f"**Recommendation: start with '{best['description']}' ({best['scenario']}).** In the simulation it "
        f"avoids {best['delayed_avoided']:,} delayed orders a month (95% range {unc(r, best['scenario'])}) while "
        f"changing how only {best['orders_affected']:,} orders are handled: {best['avoided_per_100_affected']:.1f} "
        f"delays avoided per 100 orders changed, the best ratio of any scenario at the {r['threshold']:g}-min threshold.",
        "",
        biggest_sentence(big, best),
        "",
        sensitivity_sentence(r, best["scenario"]),
        "",
        "**Trial design**",
        f"- Metric: delay rate (> {r['threshold']:g} min) of New-worker orders in Zone C.",
        f"- Baseline: {pct(full['baseline_rate'])} (observed, August 2026).",
        f"- Expected with support: {pct(full['target_rate'])}, the observed delay rate of Experienced workers in "
        f"Zone C (the S1 simulation gives {pct(r['s1_check']['simulated_new_c_rate'])}, which is not used because "
        f"scaling recorded times also shrinks their random noise). Conservative target (half the improvement): "
        f"{pct(half['target_rate'])}.",
        f"- Sample size (two-proportion test, 80% power, alpha 0.05): {full['orders_per_group']:,} orders per group "
        f"for the expected effect, {half['orders_per_group']:,} per group for the conservative effect.",
        f"- About {half['new_zone_c_orders_per_day']:.0f} New-worker Zone C orders arrive per day, so with two arms "
        f"(support on / off) the conservative case needs about {int(half['days_needed'])} day(s). Because orders from "
        f"the same worker and shift are not independent, run for at least {days} days so every shift and weekday is "
        "covered several times, and randomise by worker-shift rather than by order.",
    ]
    return "\n".join(lines)


def sections(r: dict) -> dict:
    """All generated Markdown sections, keyed by README block name."""
    return {"key_results": section_key_results(r), "describe": section_describe(r),
            "root_cause": section_root_cause(r), "models": section_models(r),
            "validation": section_validation(r), "scenarios": section_scenarios(r),
            "sensitivity": section_sensitivity(r), "recommendation": section_recommendation(r)}

# ---------------------------------------------------------------- writers


def build_report(r: dict) -> str:
    """Assemble the full outputs/report.md text."""
    s = sections(r)
    validation_rows = md_table(r["validation_data"])
    return "\n\n".join([
        "# Warehouse operations analysis: generated report",
        f"_Generated by `run_analysis.py` (threshold {r['threshold']:g} min, seed {r['seed']}, "
        f"{r['iterations']} bootstrap iterations). Every number below is computed from the data._",
        "## Key results", s["key_results"],
        "## 1. Data validation", validation_rows,
        "## 2. Description", s["describe"],
        "## 3. Root cause", s["root_cause"],
        "## 4. Separating the effects", s["models"],
        "## 4e. Validation against the known true process", s["validation"],
        "## 5. Scenario simulation", s["scenarios"],
        "### Threshold sensitivity", s["sensitivity"],
        "## 6. Recommendation and trial design", s["recommendation"],
        "## Limitations",
        "Synthetic data with built-in patterns (the project demonstrates the method, not real warehouse behaviour); "
        "one month only; no arrival times or staffing levels, so queues and throughput are not modelled; "
        "observational correlations are not proof of cause outside a controlled trial.",
    ]) + "\n"


def update_readme(readme: Path, blocks: dict) -> None:
    """Replace each <!-- BEGIN:name --> ... <!-- END:name --> block in the README."""
    text = readme.read_text(encoding="utf-8")
    for name, body in blocks.items():
        pattern = re.compile(rf"(<!-- BEGIN:{name} -->).*?(<!-- END:{name} -->)", re.S)
        text = pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", text)
    readme.write_text(text, encoding="utf-8")
