"""Loading, validating and cleaning the order data.

The order matters: load the raw file, check it (and stop if anything is
wrong), then add the helper columns the rest of the analysis uses.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

REQUIRED_COLUMNS = [
    "Order Id", "Date", "Shift", "Zone", "Product Category", "Items", "Priority",
    "Pick Time", "Pack Time", "Processing Time", "Error", "Worker Experience",
    "Orders in Shift",
]


class DataValidationError(ValueError):
    """Raised when the data fails one or more validation checks."""


def load_data(path: str | Path = config.RAW_DATA) -> pd.DataFrame:
    """Read the raw orders CSV exactly as exported from Excel.

    Excel adds an invisible byte-order mark to the first column name, so the
    file is read with 'utf-8-sig' and column names are stripped of spaces.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    return df


def _parse_dates(dates: pd.Series) -> pd.Series:
    """Turn dd/mm/yyyy text into real dates; already-parsed dates pass through.

    Anything that cannot be read becomes NaT, which the date check then flags.
    """
    if pd.api.types.is_datetime64_any_dtype(dates):
        return dates
    return pd.to_datetime(dates, format="%d/%m/%Y", errors="coerce")


def _label_issues(values: pd.Series, allowed) -> int:
    """Count values that are not in the list of expected labels (blanks excluded,
    because blanks are already counted by the missing-value check)."""
    present = values.dropna()
    return int((~present.isin(allowed)).sum())


def _check_counts(df: pd.DataFrame) -> dict:
    """Run every data-quality rule and return {check name: number of issues}.

    Numeric rules use NaN-safe comparisons so a missing value is reported once
    (by the missing-value check) rather than breaking the other checks.
    """
    dates = _parse_dates(df["Date"])
    in_month = (dates.dt.year == config.DATA_YEAR) & (dates.dt.month == config.DATA_MONTH)
    gap = (df["Processing Time"] - (df["Pick Time"] + df["Pack Time"])).abs()
    return {
        "Missing values (cells)": int(df[REQUIRED_COLUMNS].isna().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
        "Duplicate Order Ids": int(df["Order Id"].dropna().duplicated().sum()),
        "Pick or Pack time <= 0": int(((df["Pick Time"] <= 0) | (df["Pack Time"] <= 0)).sum()),
        "Items outside 1-10": int(((df["Items"] < 1) | (df["Items"] > 10)).sum()),
        "Dates outside August 2026 or unreadable": int((~in_month & df["Date"].notna()).sum()),
        f"Processing != Pick + Pack (tolerance {config.SUM_TOLERANCE})":
            int((gap > config.SUM_TOLERANCE).sum()),
        "Unexpected Zone labels": _label_issues(df["Zone"], config.ZONES),
        "Unexpected Shift labels": _label_issues(df["Shift"], config.SHIFTS),
        "Unexpected Worker Experience labels": _label_issues(df["Worker Experience"], config.EXPERIENCE),
        "Unexpected Error labels": _label_issues(df["Error"], config.ERROR_LABELS),
    }


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Check the raw data and return a validation table.

    Columns: check, issues_found, result ('PASS' or 'FAIL'). The table is saved
    as evidence that the data was checked before any number was calculated.
    """
    counts = _check_counts(df)
    table = pd.DataFrame({"check": list(counts), "issues_found": list(counts.values())})
    table["result"] = np.where(table["issues_found"] == 0, "PASS", "FAIL")
    return table


def assert_valid(validation: pd.DataFrame) -> None:
    """Stop the analysis with a clear message if any validation check failed.

    Carrying on with bad data would make every later number untrustworthy.
    """
    failed = validation[validation["result"] == "FAIL"]
    if not failed.empty:
        lines = [f"  - {r.check}: {r.issues_found} issue(s)" for r in failed.itertuples()]
        raise DataValidationError("Data validation failed:\n" + "\n".join(lines))


def clean_data(df: pd.DataFrame, threshold: float = config.DELAY_THRESHOLD) -> pd.DataFrame:
    """Return a copy of validated data with typed dates and analysis flags.

    Adds Delayed Flag (Processing Time > threshold), Error Flag, New Worker,
    Evening and Workload Band. The input dataframe is not modified.
    """
    out = df.copy()
    out["Date"] = _parse_dates(out["Date"])
    out["Delayed Flag"] = (out["Processing Time"] > threshold).astype(int)
    out["Error Flag"] = (out["Error"] == "Yes").astype(int)
    out["New Worker"] = (out["Worker Experience"] == "New").astype(int)
    out["Evening"] = (out["Shift"] == "Evening").astype(int)
    out["Workload Band"] = pd.cut(
        out["Orders in Shift"], bins=list(config.WORKLOAD_BINS),
        labels=list(config.WORKLOAD_LABELS), ordered=True,
    )
    return out


def load_clean(path: str | Path = config.RAW_DATA,
               threshold: float = config.DELAY_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate (stopping on failure) and clean in one call.

    Returns (clean dataframe, validation table). Handy for the notebook and tests.
    """
    raw = load_data(path)
    validation = validate_data(raw)
    assert_valid(validation)
    return clean_data(raw, threshold), validation
