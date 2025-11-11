"""
Purpose
-------
Tests for the general_data_notebooks_utils module.

Key behaviors
-------------
- Verify that connect_with_sqlalchemy constructs a SQLAlchemy Engine using
  environment variables and includes the UTC timezone option.
- Verify that split_windows correctly splits validity windows for configured
  tickers and removes rows whose filed_at timestamps are inconsistent with
  the pre/post-split segments.

Conventions
-----------
- Environment variables are patched via monkeypatch in tests that exercise
  connect_with_sqlalchemy to avoid depending on the real environment.
- WINDOW_SPLIT_PAIRS is patched on the target module for split_windows tests
  so that behavior is controlled and independent of production configuration.

Downstream usage
----------------
Run this test module with pytest (locally and in CI) to guard refactors of
the database connection helper and the curated window-splitting logic used
by data notebooks.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import psycopg2.extras
import pytest
from sqlalchemy.engine import Engine

from notebooks_utils.data_notebooks_utils import general_data_notebooks_utils
from notebooks_utils.data_notebooks_utils.general_data_notebooks_config import (
    WindowSplit,
)


def test_connect_with_sqlalchemy_uses_env_and_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Construct an Engine from environment variables with UTC timezone option.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch environment variables and the underlying
        SQLAlchemy create_engine function.

    Returns
    -------
    None
        The test passes if:
        - create_engine is called exactly once with the expected URL, and
        - connect_with_sqlalchemy returns the Engine instance created by
          create_engine.

    Raises
    ------
    AssertionError
        If the constructed URL does not reflect the patched environment
        variables or if create_engine is not called as expected.

    Notes
    -----
    - The test does not open any real database connections; create_engine is
      mocked to return a sentinel Engine-like object.
    """
    monkeypatch.setenv("POSTGRES_USER", "test_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_pass")
    monkeypatch.setenv("DB_HOST", "test_host")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "test_db")

    captured_urls: List[str] = []

    class DummyEngine:
        pass

    dummy_engine = DummyEngine()

    def fake_create_engine(url: str) -> Engine:  # type: ignore[override]
        captured_urls.append(url)
        # We pretend this is an Engine for the purposes of the test.
        return dummy_engine  # type: ignore[return-value]

    monkeypatch.setattr(
        general_data_notebooks_utils.sa,
        "create_engine",
        fake_create_engine,
        raising=True,
    )

    engine = general_data_notebooks_utils.connect_with_sqlalchemy()
    assert engine is dummy_engine
    assert len(captured_urls) == 1
    url = captured_urls[0]
    assert url.startswith("postgresql://test_user:test_pass@test_host:5433/test_db")
    assert "options=-c timezone=utc" in url


def test_split_windows_splits_and_filters_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Split windows and filter rows according to a single synthetic WindowSplit.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch WINDOW_SPLIT_PAIRS on the target module
        so that split_windows behavior is tested against a controlled split
        configuration.

    Returns
    -------
    None
        The test passes if:
        - validity_window is split into [lower, split_date) for pre-split CIK,
          and [split_date, upper) for post-split CIK.
        - Rows whose filed_at timestamps do not match the pre/post side are
          removed.
        - Rows not matching the split ticker/CIKs are left unchanged.

    Raises
    ------
    AssertionError
        If the resulting DataFrame does not contain the expected rows and
        validity windows after splitting.

    Notes
    -----
    - The synthetic configuration covers:
      - One ticker ('AAA') with pre- and post-split CIKs.
      - One non-matching ticker ('BBB') that should be unaffected.
    """
    # Synthetic split configuration.
    validity = psycopg2.extras.DateRange(
        lower=pd.Timestamp("2020-01-01"),
        upper=pd.Timestamp("2020-12-31"),
        bounds="[)",
    )
    split_date = pd.Timestamp("2020-06-01")

    window_split = WindowSplit(
        ticker="AAA",
        validity_window=validity,
        split_date=split_date,
        pre_split_cik="C_PRE",
        post_split_cik="C_POST",
    )

    monkeypatch.setattr(
        general_data_notebooks_utils,
        "WINDOW_SPLIT_PAIRS",
        [window_split],
        raising=True,
    )

    # Evidence:
    # - AAA / C_PRE before split_date -> keep, pre window.
    # - AAA / C_PRE after split_date  -> drop.
    # - AAA / C_POST before split     -> drop.
    # - AAA / C_POST after split      -> keep, post window.
    # - BBB / C_PRE                   -> unaffected.
    evidence_df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "AAA", "BBB"],
            "candidate_cik": ["C_PRE", "C_PRE", "C_POST", "C_POST", "C_PRE"],
            "filed_at": pd.to_datetime(
                [
                    "2020-05-01",
                    "2020-07-01",
                    "2020-05-15",
                    "2020-06-10",
                    "2020-07-01",
                ]
            ),
            "validity_window": [
                validity,
                validity,
                validity,
                validity,
                psycopg2.extras.DateRange(
                    lower=pd.Timestamp("2019-01-01"),
                    upper=pd.Timestamp("2019-12-31"),
                    bounds="[)",
                ),
            ],
        }
    )

    result = general_data_notebooks_utils.split_windows(evidence_df)

    # We expect three rows: AAA/C_PRE (pre), AAA/C_POST (post), BBB/C_PRE (unchanged).
    assert len(result) == 3

    # AAA / C_PRE row
    aaa_pre = result[(result["ticker"] == "AAA") & (result["candidate_cik"] == "C_PRE")]
    assert len(aaa_pre) == 1
    pre_range = aaa_pre["validity_window"].iloc[0]
    assert isinstance(pre_range, psycopg2.extras.DateRange)
    assert pre_range.lower == validity.lower
    assert pre_range.upper == split_date.date()

    # AAA / C_POST row
    aaa_post = result[(result["ticker"] == "AAA") & (result["candidate_cik"] == "C_POST")]
    assert len(aaa_post) == 1
    post_range = aaa_post["validity_window"].iloc[0]
    assert isinstance(post_range, psycopg2.extras.DateRange)
    assert post_range.lower == split_date.date()
    assert post_range.upper == validity.upper

    # BBB row should be unaffected (including its validity_window).
    bbb = result[(result["ticker"] == "BBB") & (result["candidate_cik"] == "C_PRE")]
    assert len(bbb) == 1
    original_bbb_range = evidence_df.loc[
        (evidence_df["ticker"] == "BBB") & (evidence_df["candidate_cik"] == "C_PRE"),
        "validity_window",
    ].iloc[0]
    assert bbb["validity_window"].iloc[0] == original_bbb_range


def test_split_windows_no_matching_ticker_returns_same_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure split_windows leaves the DataFrame unchanged when no tickers match.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch WINDOW_SPLIT_PAIRS on the target module
        to a configuration that does not match any rows in the input frame.

    Returns
    -------
    None
        The test passes if the output DataFrame has the same number of rows
        and identical tickers/CIKs as the input, indicating no unintended
        filtering or modification.

    Raises
    ------
    AssertionError
        If rows are added or removed despite there being no matching splits.

    Notes
    -----
    - This test guards against regressions where the filtering logic might
      over-remove rows due to a bug in the identity_mask construction.
    """
    validity = psycopg2.extras.DateRange(
        lower=pd.Timestamp("2020-01-01"),
        upper=pd.Timestamp("2020-12-31"),
        bounds="[)",
    )

    # Split config for a ticker that does not exist in the evidence.
    window_split = WindowSplit(
        ticker="ZZZ",
        validity_window=validity,
        split_date=pd.Timestamp("2020-06-01"),
        pre_split_cik="C_PRE",
        post_split_cik="C_POST",
    )

    monkeypatch.setattr(
        general_data_notebooks_utils,
        "WINDOW_SPLIT_PAIRS",
        [window_split],
        raising=True,
    )

    evidence_df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "candidate_cik": ["C1", "C2"],
            "filed_at": pd.to_datetime(["2020-03-01", "2020-09-01"]),
            "validity_window": [validity, validity],
        }
    )

    result = general_data_notebooks_utils.split_windows(evidence_df)

    # No rows should be removed or added.
    assert len(result) == len(evidence_df)
    assert result["ticker"].tolist() == evidence_df["ticker"].tolist()
    assert result["candidate_cik"].tolist() == evidence_df["candidate_cik"].tolist()
