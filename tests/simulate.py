"""A small simulator of the documented data-generating process.

Used by the parameter-recovery test: if we generate data with KNOWN
parameters and the model gets them back, we can trust it on the real data.
The pack-time intercept/slope and noise levels are not documented, so the
simulator picks its own values; the test only checks what it knows.
"""

import numpy as np
import pandas as pd

TRUE = {"base": {"A": 4.0, "B": 5.0, "C": 7.0, "D": 5.0}, "k": 0.3, "m_evening": 1.15, "m_new": 1.25,
        "pack_a": 3.0, "pack_b": 0.2, "p_new": 1.15,
        "error": {"baseline": 0.02, "zone_C": 0.04, "evening": 0.02, "new": 0.03}}


def _times(zone, items, evening, new, rng):
    """Pick and pack times from the documented formulas plus random noise."""
    base = pd.Series(zone).map(TRUE["base"]).to_numpy()
    pick = (base + TRUE["k"] * items) * TRUE["m_evening"] ** evening * TRUE["m_new"] ** new
    pick = np.round(np.maximum(pick + rng.normal(0, 0.8, len(zone)), 0.5), 1)
    pack = (TRUE["pack_a"] + TRUE["pack_b"] * items) * TRUE["p_new"] ** new
    pack = np.round(np.maximum(pack + rng.normal(0, 0.4, len(zone)), 0.5), 1)
    return pick, pack


def simulate_orders(n: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Generate `n` orders in the same format as data/raw_data.csv."""
    rng = np.random.default_rng(seed)
    zone = rng.choice(list("ABCD"), n)
    shift = rng.choice(["Morning", "Afternoon", "Evening"], n, p=[0.3, 0.3, 0.4])
    items = rng.integers(1, 11, n)
    evening, new = (shift == "Evening").astype(float), (rng.random(n) < 0.3).astype(float)
    pick, pack = _times(zone, items, evening, new, rng)
    e = TRUE["error"]
    p_err = e["baseline"] + e["zone_C"] * (zone == "C") + e["evening"] * evening + e["new"] * new
    days = rng.integers(1, 32, n)
    df = pd.DataFrame({
        "Order Id": [f"S{i:05d}" for i in range(n)],
        "Date": [f"{d:02d}/08/2026" for d in days], "Shift": shift, "Zone": zone,
        "Product Category": rng.choice(["Clothing", "Electronics", "Home"], n), "Items": items,
        "Priority": rng.choice(["Priority", "Standard"], n, p=[0.25, 0.75]),
        "Pick Time": pick, "Pack Time": pack, "Processing Time": np.round(pick + pack, 1),
        "Error": np.where(rng.random(n) < p_err, "Yes", "No"),
        "Worker Experience": np.where(new == 1, "New", "Experienced"),
    })
    df["Orders in Shift"] = df.groupby(["Date", "Shift"])["Order Id"].transform("size")
    return df
