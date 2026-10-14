"""Validation must catch bad data and pass good data."""

import numpy as np
import pandas as pd
import pytest

from warehouse import data


def good_rows(n: int = 4) -> pd.DataFrame:
    """A tiny, valid dataframe in the raw-file format."""
    return pd.DataFrame({
        "Order Id": [f"O{i}" for i in range(n)], "Date": ["01/08/2026"] * n,
        "Shift": ["Morning"] * n, "Zone": ["A"] * n, "Product Category": ["Home"] * n,
        "Items": [3] * n, "Priority": ["Standard"] * n, "Pick Time": [5.0] * n,
        "Pack Time": [3.0] * n, "Processing Time": [8.0] * n, "Error": ["No"] * n,
        "Worker Experience": ["New"] * n, "Orders in Shift": [100] * n,
    })


def issues(table: pd.DataFrame, check_start: str) -> int:
    """Issues found by the check whose name starts with `check_start`."""
    return int(table.loc[table["check"].str.startswith(check_start), "issues_found"].iloc[0])


def test_clean_rows_pass():
    table = data.validate_data(good_rows())
    assert (table["result"] == "PASS").all()


def test_bad_rows_are_flagged():
    df = good_rows(5)
    df.loc[0, "Zone"] = np.nan                      # missing value
    df.loc[1, "Order Id"] = "O2"                    # duplicate Order Id (O2 exists)
    df.loc[3, "Pick Time"] = -1.0                   # negative pick time ...
    df.loc[3, "Processing Time"] = 2.0              # ... with a consistent sum
    df.loc[4, "Processing Time"] = 9.5              # Processing != Pick + Pack
    table = data.validate_data(df)
    assert issues(table, "Missing values") == 1
    assert issues(table, "Duplicate Order Ids") == 1
    assert issues(table, "Pick or Pack time <= 0") == 1
    assert issues(table, "Processing != Pick + Pack") == 1
    assert issues(table, "Unexpected Zone labels") == 0  # the blank is counted once, as missing
    with pytest.raises(data.DataValidationError):
        data.assert_valid(table)


def test_other_checks():
    df = good_rows(4)
    df.loc[0, "Items"] = 11
    df.loc[1, "Date"] = "01/09/2026"
    df.loc[2, "Shift"] = "Night"
    table = data.validate_data(df)
    assert issues(table, "Items outside") == 1
    assert issues(table, "Dates outside") == 1
    assert issues(table, "Unexpected Shift") == 1


def test_real_data_passes_and_flags_use_threshold(real_df):
    assert len(real_df) == 10000
    assert real_df["Delayed Flag"].sum() == (real_df["Processing Time"] > 15).sum()
    strict = data.clean_data(data.load_data(), threshold=12)
    assert strict["Delayed Flag"].sum() > real_df["Delayed Flag"].sum()
