"""Descriptive metrics: headline KPIs and group-by summaries.

Two rates are reported for every group and they answer different questions:

* delay rate        = delayed orders in the group / orders in the group
                      ("how likely is an order from this group to be late?")
* share of delays   = delayed orders in the group / ALL delayed orders
                      ("how much of the total delay problem sits in this group?")

A small group can have a high delay rate but a small share of delays; a big
group can have a modest rate but a large share. Both are needed to prioritise.
"""

import pandas as pd


def p90(values: pd.Series) -> float:
    """90th percentile: the time that 9 in 10 orders finish within."""
    return float(values.quantile(0.90))


def kpis(df: pd.DataFrame) -> dict:
    """Headline numbers for the whole month.

    Uses the Delayed Flag already on the dataframe, so the delay definition is
    set once (in clean_data or a scenario) and never re-implemented here.
    """
    return {
        "orders": int(len(df)),
        "delayed_orders": int(df["Delayed Flag"].sum()),
        "delay_rate": float(df["Delayed Flag"].mean()),
        "error_rate": float(df["Error Flag"].mean()),
        "mean_processing": float(df["Processing Time"].mean()),
        "median_processing": float(df["Processing Time"].median()),
        "p90_processing": p90(df["Processing Time"]),
        "mean_pick": float(df["Pick Time"].mean()),
        "mean_pack": float(df["Pack Time"].mean()),
    }


def summarise_by(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    """Summarise orders, delays, processing time and errors for each group.

    Returns one row per group with: orders, share_of_orders, delayed_orders,
    delay_rate, share_of_delays, mean/median/p90 processing time and error_rate.
    Rates are proportions (0-1), not percentages, so they can be reused in
    further calculations without dividing by 100.
    """
    cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    table = df.groupby(cols, observed=True).agg(
        orders=("Order Id", "size"),
        delayed_orders=("Delayed Flag", "sum"),
        delay_rate=("Delayed Flag", "mean"),
        mean_processing=("Processing Time", "mean"),
        median_processing=("Processing Time", "median"),
        p90_processing=("Processing Time", p90),
        error_rate=("Error Flag", "mean"),
    )
    total_delayed = df["Delayed Flag"].sum()
    table.insert(1, "share_of_orders", table["orders"] / len(df))
    table.insert(4, "share_of_delays",
                 table["delayed_orders"] / total_delayed if total_delayed else 0.0)
    return table.reset_index()
