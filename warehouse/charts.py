"""Every chart in the project, drawn from already-computed tables.

Charts never recalculate results themselves: they only draw tables produced
by metrics.py, model.py and scenarios.py, so a chart can never disagree with
the saved tables or the report. One highlight colour marks the key group.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed (works on CI servers)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

C = config.COLOURS


def apply_style() -> None:
    """Set one consistent, readable look for every chart."""
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"], "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.labelsize": 11, "axes.edgecolor": C["muted"],
        "axes.labelcolor": C["text"], "xtick.color": C["muted"], "ytick.color": C["muted"],
        "text.color": C["text"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False,
    })


def save(fig, out_dir: Path, name: str) -> Path:
    """Save a figure at the configured dpi and close it to free memory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _zone_colours(zones) -> list[str]:
    """Highlight colour for the key zone, base colour for the rest."""
    return [C["highlight"] if z == config.KEY_ZONE else C["base"] for z in zones]


def delay_rate_by_zone(zone: pd.DataFrame, out_dir: Path) -> Path:
    """Chart 1: delay rate per zone with Wilson 95% CI error bars."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rate = zone["delay_rate"] * 100
    err = np.vstack([rate - zone["delay_rate_ci_low"] * 100, zone["delay_rate_ci_high"] * 100 - rate])
    ax.bar(zone["Zone"], rate, color=_zone_colours(zone["Zone"]), width=0.6,
           yerr=err, capsize=5, error_kw={"ecolor": C["text"], "lw": 1.2})
    for x, v in zip(zone["Zone"], rate):
        ax.annotate(f"{v:.1f}%", (x, v), xytext=(14, 4), textcoords="offset points", fontsize=10)
    ax.set(title="Delay rate by zone (95% Wilson CI)", xlabel="Zone",
           ylabel=f"Orders over {config.DELAY_THRESHOLD:g} min (%)")
    return save(fig, out_dir, "01_delay_rate_by_zone.png")


def share_orders_vs_delays(zone: pd.DataFrame, out_dir: Path) -> Path:
    """Chart 2: each zone's share of all orders next to its share of all delays."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(zone))
    delay_colours = [C["highlight"] if z == config.KEY_ZONE else C["dark"] for z in zone["Zone"]]
    ax.bar(x - 0.2, zone["share_of_orders"] * 100, 0.38, color=C["neutral"], label="Share of all orders")
    ax.bar(x + 0.2, zone["share_of_delays"] * 100, 0.38, color=delay_colours,
           label="Share of all delayed orders (Zone C highlighted)")
    for xi, (o, d) in enumerate(zip(zone["share_of_orders"], zone["share_of_delays"])):
        ax.text(xi - 0.2, o * 100 + 1, f"{o:.0%}", ha="center", fontsize=9, color=C["muted"])
        ax.text(xi + 0.2, d * 100 + 1, f"{d:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x, zone["Zone"])
    ax.set(title="Share of orders vs share of delays", xlabel="Zone", ylabel="Share (%)")
    ax.legend(loc="upper left")
    return save(fig, out_dir, "02_share_orders_vs_delays.png")


def zone_experience_heatmap(table: pd.DataFrame, out_dir: Path) -> Path:
    """Chart 3: delay rate for every Zone x Worker Experience cell, annotated."""
    pivot = table.pivot(index="Worker Experience", columns="Zone", values="delay_rate").reindex(config.EXPERIENCE)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    im = ax.imshow(pivot.to_numpy() * 100, cmap=config.HEATMAP_CMAP, vmin=0, aspect="auto")
    for (i, j), v in np.ndenumerate(pivot.to_numpy()):
        n = table[(table["Worker Experience"] == pivot.index[i]) & (table["Zone"] == pivot.columns[j])]["orders"].iloc[0]
        ax.text(j, i, f"{v:.0%}\n(n={n})", ha="center", va="center", fontsize=10,
                color="white" if v > 0.35 else C["text"])
    i, j = pivot.index.get_loc("New"), pivot.columns.get_loc(config.KEY_ZONE)
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=C["highlight"], lw=3))
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.grid(False)
    ax.set(title="Delay rate by zone and worker experience", xlabel="Zone", ylabel="Worker experience")
    fig.colorbar(im, ax=ax, label="Delay rate (%)")
    return save(fig, out_dir, "03_zone_experience_heatmap.png")


def processing_time_by_zone(df: pd.DataFrame, out_dir: Path) -> Path:
    """Chart 4: processing-time distributions by zone with the delay threshold."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    data = [df.loc[df["Zone"] == z, "Processing Time"] for z in config.ZONES]
    box = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": C["text"]},
                     flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    ax.set_xticks(range(1, len(config.ZONES) + 1), config.ZONES)
    for patch, colour in zip(box["boxes"], _zone_colours(config.ZONES)):
        patch.set_facecolor(colour)
    ax.axhline(config.DELAY_THRESHOLD, color=C["text"], ls="--", lw=1.2)
    ax.text(4.45, config.DELAY_THRESHOLD + 0.3, f"{config.DELAY_THRESHOLD:g}-min delay threshold",
            ha="right", fontsize=9)
    ax.set(title="Processing time by zone", xlabel="Zone", ylabel="Processing time (min)")
    return save(fig, out_dir, "04_processing_time_by_zone.png")


def workload_confounding(per_shift: pd.DataFrame, out_dir: Path) -> Path:
    """Chart 5: each dot is one date-shift; within a shift the slope is ~flat.

    The overall upward trend comes from evening shifts being both busier and
    slower, not from workload itself.
    """
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for shift in config.SHIFTS:
        part = per_shift[per_shift["Shift"] == shift]
        ax.scatter(part["Orders in Shift"], part["delay_rate"] * 100, s=28, alpha=0.8,
                   color=config.SHIFT_COLOURS[shift], label=shift, edgecolor=C["surface"], lw=0.8)
        slope, intercept = np.polyfit(part["Orders in Shift"], part["delay_rate"] * 100, 1)
        xs = np.linspace(part["Orders in Shift"].min(), part["Orders in Shift"].max(), 10)
        ax.plot(xs, intercept + slope * xs, color=config.SHIFT_COLOURS[shift], lw=2)
    ax.set(title="Delay rate vs workload, by shift (one dot = one shift on one day)",
           xlabel="Orders in shift", ylabel="Delay rate in that shift (%)")
    ax.legend(title="Shift", loc="upper left")
    return save(fig, out_dir, "05_workload_confounding.png")


def _forest(ax, rows: pd.DataFrame, title: str, xlabel: str, relative: bool = False) -> None:
    """Draw estimates with CI lines and the true value as a black tick.

    With relative=True the values are shown as % difference from the truth,
    which makes very tight CIs on very different scales comparable.
    """
    rows = rows.reset_index(drop=True)
    scale = rows["true_value"] if relative else 1.0
    shift = rows["true_value"] if relative else 0.0
    est, low, high, truth = [(rows[c] - shift) / scale * (100 if relative else 1)
                             for c in ("estimate", "ci_low", "ci_high", "true_value")]
    y = np.arange(len(rows))[::-1]
    colours = [C["dark"] if ok else C["highlight"] for ok in rows["contains_truth"]]
    ax.hlines(y, low, high, color=colours, lw=2.5)
    ax.scatter(est, y, color=colours, s=40, zorder=3, label="Estimate with 95% CI")
    ax.scatter(truth, y, marker="|", s=300, color=C["truth"], lw=2, zorder=4, label="True value")
    labels = [f"{p} (true {t:g})" for p, t in zip(rows["parameter"], rows["true_value"])]
    ax.set_yticks(y, labels)
    ax.set(title=title, xlabel=xlabel)


def validation_chart(validation: pd.DataFrame, out_dir: Path) -> Path:
    """Chart 6: true value vs estimate (with 95% CI) for every parameter.

    Blue = the CI contains the truth, orange = it does not.
    """
    v = validation
    panels = [
        (v[v["section"].isin(["Pick time model", "Pack time model"])], "Time models",
         "Estimate vs true value (% difference)", True),
        (v[v["section"] == "Error model (LPM)"], "Error model", "Effect on error rate (percentage points)", False),
        (v[v["section"] == "Pick time: no-effect checks"], "Pick time: no-effect checks",
         "Effect on pick time (%)", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    for ax, (rows, title, xlabel, relative) in zip(axes, panels):
        _forest(ax, rows, title, xlabel, relative)
    handles = [Line2D([], [], color=C["dark"], marker="o", lw=2.5, label="Estimate, 95% CI contains truth"),
               Line2D([], [], color=C["highlight"], marker="o", lw=2.5, label="Estimate, 95% CI misses truth"),
               Line2D([], [], color=C["truth"], marker="|", ms=14, mew=2, lw=0, label="True value")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Model validation: estimates vs the known true process", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save(fig, out_dir, "06_validation_true_vs_estimate.png")


def scenario_comparison(table: pd.DataFrame, best: str, out_dir: Path) -> Path:
    """Chart 7: delayed orders avoided per scenario with the bootstrap 95% range."""
    t = table[table["scenario"] != "S0"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colours = [C["highlight"] if s == best else C["base"] for s in t["scenario"]]
    err = np.vstack([t["delayed_avoided"] - t["avoided_low"], t["avoided_high"] - t["delayed_avoided"]])
    ax.barh(t["scenario"], t["delayed_avoided"], color=colours, xerr=err, capsize=4,
            error_kw={"ecolor": C["text"], "lw": 1.1})
    for y, (v, hi, n) in enumerate(zip(t["delayed_avoided"], t["avoided_high"], t["orders_affected"])):
        ax.text(hi + 12, y, f"{v:,} avoided ({n:,} orders changed)", va="center", fontsize=9)
    ax.set_xlim(0, t["avoided_high"].max() * 1.45)
    ax.set(title="Delayed orders avoided per month (95% range)", xlabel="Delayed orders avoided vs baseline")
    ax.grid(axis="y", visible=False)
    return save(fig, out_dir, "07_scenario_comparison.png")


def s3_sensitivity(table: pd.DataFrame, out_dir: Path) -> Path:
    """Chart 8: Zone C delay rate as the layout saving (delta) grows."""
    s3 = table[table["scenario"].str.startswith("S3")]
    deltas = [0.0] + [float(s.split("(")[1].split()[0]) for s in s3["scenario"]]
    rates = [table.loc[table["scenario"] == "S0", "zone_c_delay_rate"].iloc[0]] + list(s3["zone_c_delay_rate"])
    s1 = table.loc[table["scenario"] == "S1", "zone_c_delay_rate"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(deltas, np.array(rates) * 100, marker="o", color=C["dark"], lw=2, label="S3: layout change")
    ax.axhline(s1 * 100, color=C["highlight"], ls="--", lw=1.5, label="S1: support new workers")
    for d, r in zip(deltas, rates):
        ax.annotate(f"{r:.1%}", (d, r * 100), xytext=(5, 6), textcoords="offset points", fontsize=9)
    ax.set(title="Zone C delay rate vs layout improvement",
           xlabel="Reduction in Zone C base pick time, delta (min)", ylabel="Zone C delay rate (%)")
    ax.legend()
    return save(fig, out_dir, "08_s3_delta_sensitivity.png")


def threshold_sensitivity(sens: pd.DataFrame, out_dir: Path) -> Path:
    """Chart 9: scenario ranking (avoided per 100 changed orders) at each threshold."""
    thresholds = sorted(sens["threshold"].unique())
    fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 4.8), sharey=False)
    for ax, t in zip(np.atleast_1d(axes), thresholds):
        part = sens[sens["threshold"] == t].sort_values("avoided_per_100_affected")
        colours = [C["highlight"] if r == 1 else C["base"] for r in part["rank"]]
        ax.barh(part["scenario"], part["avoided_per_100_affected"], color=colours)
        ax.axvline(0, color=C["muted"], lw=1)
        ax.set(title=f"Threshold {t:g} min", xlabel="Delays avoided per 100 orders changed")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Is the best scenario the same at every delay threshold? (orange = rank 1)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save(fig, out_dir, "09_threshold_sensitivity.png")
