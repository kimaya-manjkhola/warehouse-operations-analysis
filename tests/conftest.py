"""Shared test fixtures: the real data, a simulated dataset and one pipeline run."""

import pandas as pd
import pytest

import run_analysis
from tests.simulate import simulate_orders
from warehouse import config, data


@pytest.fixture(scope="session")
def real_df() -> pd.DataFrame:
    """The real (synthetic Excel) data, validated and cleaned."""
    df, _ = data.load_clean(config.RAW_DATA)
    return df


@pytest.fixture(scope="session")
def simulated_df() -> pd.DataFrame:
    """20,000 orders generated with known true parameters."""
    return data.clean_data(simulate_orders(n=20000, seed=7))


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory) -> dict:
    """One full pipeline run into a temporary folder (few bootstrap repeats for speed)."""
    out = tmp_path_factory.mktemp("outputs")
    return run_analysis.run(iterations=30, out_dir=out)
