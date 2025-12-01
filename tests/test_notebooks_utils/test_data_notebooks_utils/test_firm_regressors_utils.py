"""
Purpose
-------
Exercise the data-fetching, alignment, and feature-construction logic in
`firm_regressors_utils`. The tests focus on SQL extraction, HTTP-based
fundamentals and OHLCV ingestion, quarterly filtering, as-of alignment, and
the no-look-ahead return and control calculations.

Key behaviors
-------------
- Validate that `extract_active_firms` returns a DataFrame built from the
  configured SQL query and a SQLAlchemy engine.
- Verify that fundamentals and returns builders aggregate per-firm records
  from HTTP responses using ticker alias mappings and validity windows.
- Check that quarterly and OHLCV helpers filter by date ranges and map
  payloads into the expected column layout.
- Assert that `align_fundamentals_with_returns` performs a backward
  merge-as-of within the configured tolerance window.
- Confirm that `calculate_features` computes market-cap, book-to-market,
  return decompositions, realized volatilities, and momentum signals without
  look-ahead.

Conventions
-----------
- Tests use pytest and monkeypatch network/DB dependencies; no real HTTP
  requests or database connections are exercised.
- A minimal dummy logger implements the subset of the `InfraLogger` interface
  required by the module under test.
- Constants such as `QUARTER_IN_DAYS` and `TICKER_ALIAS_MAPPING` are
  monkeypatched where necessary to keep tests deterministic.

Downstream usage
----------------
Run these tests with pytest as part of the CI pipeline to guard the
firm-level panel construction invariants prior to running the Glasserman-style
regressions.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Sequence, cast

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

from infra.logging.infra_logger import InfraLogger
from notebooks_utils.data_notebooks_utils.firm_regressors_utils import (
    firm_regressors_utils as firm_utils,
)


class DummyInfraLogger:
    """
    Purpose
    -------
    Provide a minimal stand-in for `InfraLogger` that satisfies the logging
    interface expected by the firm-regressors utilities without performing
    any I/O.

    Key behaviors
    -------------
    - Exposes `debug` and `warning` methods that accept arbitrary positional
      and keyword arguments and discard them.
    - Can be passed anywhere an `InfraLogger` instance is required in the
      firm-utils module.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Notes
    -----
    - This class is intentionally a no-op and exists only to decouple the
      tests from the concrete logging implementation.
    """

    def debug(self, *args: object, **kwargs: object) -> None:
        """
        Accept arbitrary arguments and perform no operation.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded from the caller.
        **kwargs : object
            Keyword arguments forwarded from the caller.

        Returns
        -------
        None
            The call is intentionally a no-op.
        """
        return None

    def warning(self, *args: object, **kwargs: object) -> None:
        """
        Accept arbitrary arguments and perform no operation for warnings.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded from the caller.
        **kwargs : object
            Keyword arguments forwarded from the caller.

        Returns
        -------
        None
            The call is intentionally a no-op.
        """
        return None


class DummyResponseContext:
    """
    Purpose
    -------
    Represent a minimal context manager wrapper for HTTP responses returned
    by the monkeypatched `make_request` helper in tests.

    Key behaviors
    -------------
    - Implements the context-manager protocol used by the production code.
    - Returns the wrapped payload from `json()`.

    Parameters
    ----------
    payload : object
        Arbitrary object that should be yielded as the JSON body.

    Attributes
    ----------
    _payload : object
        Stored JSON-serializable payload to return from `json()`.

    Notes
    -----
    - The implementation is deliberately simple: it does not attempt to
      emulate status codes or headers.
    """

    def __init__(self, payload: object) -> None:
        """
        Initialize the response wrapper with a JSON payload.

        Parameters
        ----------
        payload : object
            JSON-like payload to return from `json()`.

        Returns
        -------
        None
        """
        self._payload = payload

    def __enter__(self) -> "DummyResponseContext":
        """
        Enter the context manager.

        Parameters
        ----------
        None

        Returns
        -------
        DummyResponseContext
            The context manager instance itself.
        """
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> Literal[False]:
        """
        Exit the context manager and do nothing special.

        Parameters
        ----------
        exc_type : object
            Exception type, if any.
        exc_val : object
            Exception instance, if any.
        exc_tb : object
            Traceback, if any.

        Returns
        -------
        bool
            Always returns False to propagate any exceptions.
        """
        return False

    def json(self) -> object:
        """
        Return the stored JSON payload.

        Parameters
        ----------
        None

        Returns
        -------
        object
            The JSON-like payload passed at construction time.
        """
        return self._payload


@pytest.fixture
def dummy_logger() -> DummyInfraLogger:
    """
    Construct a dummy logger suitable for use in tests.

    Parameters
    ----------
    None

    Returns
    -------
    DummyInfraLogger
        A logger whose `debug` and `warning` methods ignore all input.

    Notes
    -----
    - This fixture avoids importing or configuring the real `InfraLogger`
      implementation within the test suite.
    """
    return DummyInfraLogger()


def test_extract_active_firms_uses_engine_and_returns_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `extract_active_firms` calls `connect_with_sqlalchemy`, passes
    the resulting engine into `pandas.read_sql`, and returns the resulting
    DataFrame.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Utility for patching the SQL connector and pandas read_sql.

    Returns
    -------
    None
        The test passes if the returned DataFrame matches the stubbed result.
    """

    class DummyEngine:
        """Minimal stand-in for a SQLAlchemy engine."""

    dummy_engine = DummyEngine()
    captured_args: Dict[str, object] = {}

    def fake_connect() -> sa.Engine:
        """Return a dummy engine for testing."""
        return cast(sa.Engine, dummy_engine)

    def fake_read_sql(query: str, engine: sa.Engine) -> pd.DataFrame:
        """Capture query and engine and return a small DataFrame."""
        captured_args["query"] = query
        captured_args["engine"] = engine
        return pd.DataFrame(
            {
                "ticker": ["ABC"],
                "validity_window": ["[2016-01-01,2017-01-01)"],
                "cik": ["0000000001"],
            }
        )

    monkeypatch.setattr(firm_utils, "connect_with_sqlalchemy", fake_connect)
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    result: pd.DataFrame = firm_utils.extract_active_firms()

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 1
    assert "ticker" in result.columns
    assert captured_args["engine"] is dummy_engine
    assert "FROM ticker_cik_mapping" in str(captured_args["query"])


def test_construct_fundamentals_url_builds_expected_endpoint() -> None:
    """
    Ensure that `construct_fundamentals_url` concatenates the symbol,
    filter, and API token into the expected EODHD fundamentals endpoint.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the resulting URL contains the symbol, filter, and
        API token in the right places.
    """
    # Use a simple stand-in for FUNDAMENTALS_FILTER to avoid depending on
    # specific project configuration values.
    original_filter: str = firm_utils.FUNDAMENTALS_FILTER
    try:
        firm_utils.FUNDAMENTALS_FILTER = "Highlights"
        url: str = firm_utils.construct_fundamentals_url("ABC.US", "TESTKEY")
    finally:
        firm_utils.FUNDAMENTALS_FILTER = original_filter

    assert url.startswith("https://eodhd.com/api/fundamentals/ABC.US?")
    assert "filter=Highlights" in url
    assert "api_token=TESTKEY" in url
    assert url.endswith("&fmt=json")


def test_extract_quarterly_data_filters_by_window_and_missing_fields(
    dummy_logger: DummyInfraLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate that `extract_quarterly_data` skips quarters older than one
    quarter before the validity window, ignores records with missing filing
    dates, and computes book equity and shares outstanding when available.

    Parameters
    ----------
    dummy_logger : DummyInfraLogger
        Logger stand-in whose methods are no-ops.
    monkeypatch : pytest.MonkeyPatch
        Used to patch `QUARTER_IN_DAYS` for deterministic behavior.

    Returns
    -------
    None
        The test passes if only the valid quarterly record is returned with
        the expected fields.
    """
    monkeypatch.setattr(firm_utils, "QUARTER_IN_DAYS", 90)

    quarterly_dict: Dict[str, Dict[str, Any]] = {
        # Too old relative to validity window (will be skipped).
        "2015-09-30": {
            "filing_date": "2015-10-15",
            "totalAssets": "100.0",
            "totalLiab": "40.0",
            "commonStockSharesOutstanding": "10.0",
        },
        # Missing filing_date (skipped with warning).
        "2016-05-31": {
            "totalAssets": "200.0",
            "totalLiab": "80.0",
            "commonStockSharesOutstanding": "20.0",
        },
        # Valid record inside the window.
        "2016-06-30": {
            "filing_date": "2016-07-15",
            "totalAssets": "300.0",
            "totalLiab": "100.0",
            "commonStockSharesOutstanding": "30.0",
        },
    }

    validity_start: pd.Timestamp = pd.Timestamp("2016-01-01")
    validity_end: pd.Timestamp = pd.Timestamp("2017-01-01")
    validity_window = (validity_start, validity_end)

    logger = cast(InfraLogger, dummy_logger)

    records: List[dict] = firm_utils.extract_quarterly_data(
        quarterly_dict=quarterly_dict,
        current_symbol="ABC.US",
        current_validity_window=validity_window,
        current_cik="0000000001",
        logger=logger,
    )

    assert len(records) == 1
    row: dict = records[0]
    assert row["ticker"] == "ABC.US"
    assert row["cik"] == "0000000001"
    # book_equity = totalAssets - totalLiab
    assert row["book_equity"] == pytest.approx(200.0)
    assert row["shares_outstanding"] == pytest.approx(30.0)
    assert isinstance(row["filing_date"], pd.Timestamp)
    assert row["filing_date"] == pd.Timestamp("2016-07-15")


def test_build_fundamentals_df_aggregates_records_from_active_firms(
    dummy_logger: DummyInfraLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Confirm that `build_fundamentals_df` iterates over active firms, applies
    the ticker alias mapping, calls `construct_fundamentals_url`, and
    aggregates quarterly records returned by `extract_quarterly_data`.

    Parameters
    ----------
    dummy_logger : DummyInfraLogger
        Logger instance whose calls are ignored.
    monkeypatch : pytest.MonkeyPatch
        Utility for monkeypatching alias mappings, HTTP layer, and helper
        functions.

    Returns
    -------
    None
        The test passes if the resulting DataFrame contains one row per
        record returned by the stubbed `extract_quarterly_data` helper and
        the alias mapping is respected.
    """
    validity_window_obj = SimpleNamespace(
        lower=pd.Timestamp("2016-01-01"),
        upper=pd.Timestamp("2017-01-01"),
    )

    active_firms = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "validity_window": [validity_window_obj],
            "cik": ["0000000001"],
        }
    )

    # Ensure ticker alias mapping is used (ABC -> XYZ).
    monkeypatch.setattr(
        firm_utils,
        "TICKER_ALIAS_MAPPING",
        {"ABC": "XYZ"},
    )

    called_symbols: List[str] = []

    def fake_construct(symbol: str, api_key: str) -> str:
        """Record constructed symbol and return a dummy URL."""
        called_symbols.append(symbol)
        return f"https://example/{symbol}?token={api_key}"

    def fake_make_request(url: str) -> DummyResponseContext:
        """Return a dummy response with a dict payload."""
        # The payload only needs to be a dict to exercise the main branch.
        return DummyResponseContext(payload={"dummy": "payload"})

    def fake_extract_quarterly_data(
        quarterly_dict: Dict[str, Any],
        current_symbol: str,
        current_validity_window: Any,
        current_cik: str,
        logger: InfraLogger,
    ) -> List[dict]:
        """Return a single synthetic record per firm."""
        return [
            {
                "ticker": current_symbol,
                "cik": current_cik,
                "filing_date": pd.Timestamp("2016-07-15"),
                "book_equity": 100.0,
                "shares_outstanding": 10.0,
            }
        ]

    monkeypatch.setattr(firm_utils, "construct_fundamentals_url", fake_construct)
    monkeypatch.setattr(firm_utils, "make_request", fake_make_request)
    monkeypatch.setattr(firm_utils, "extract_quarterly_data", fake_extract_quarterly_data)

    logger = cast(InfraLogger, dummy_logger)
    df: pd.DataFrame = firm_utils.build_fundamentals_df(
        active_firms=active_firms,
        api_key="TESTKEY",
        logger=logger,
        real_run=True,
    )

    # One record per active firm as returned by fake_extract_quarterly_data.
    assert df.shape[0] == 1
    assert df.loc[0, "ticker"] == "XYZ.US"
    assert df.loc[0, "cik"] == "0000000001"
    assert called_symbols == ["XYZ.US"]


def test_build_fundamentals_df_skips_non_dict_payloads(
    dummy_logger: DummyInfraLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure that `build_fundamentals_df` skips a firm when the fundamentals
    endpoint returns a non-dict JSON payload.

    Parameters
    ----------
    dummy_logger : DummyInfraLogger
        Logger instance whose warnings are ignored.
    monkeypatch : pytest.MonkeyPatch
        Used to patch `make_request` to return a non-dict payload.

    Returns
    -------
    None
        The test passes if the resulting DataFrame is empty.
    """
    validity_window_obj = SimpleNamespace(
        lower=pd.Timestamp("2016-01-01"),
        upper=pd.Timestamp("2017-01-01"),
    )

    active_firms = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "validity_window": [validity_window_obj],
            "cik": ["0000000001"],
        }
    )

    def fake_make_request(url: str) -> DummyResponseContext:
        """Return a list to trigger the non-dict branch."""
        return DummyResponseContext(payload=[{"unexpected": "list"}])

    monkeypatch.setattr(firm_utils, "make_request", fake_make_request)

    logger = cast(InfraLogger, dummy_logger)
    df: pd.DataFrame = firm_utils.build_fundamentals_df(
        active_firms=active_firms,
        api_key="TESTKEY",
        logger=logger,
        real_run=True,
    )

    assert df.empty


def test_extract_ohlcv_filters_by_validity_window() -> None:
    """
    Validate that `extract_ohlcv` only keeps records whose dates fall within
    the half-open validity window [start, end) and maps payload keys into the
    expected column names.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if only in-window records are returned and all
        expected keys are present.
    """
    ohlcv_dicts: List[Dict[str, Any]] = [
        {
            "date": "2015-12-31",
            "open": 10.0,
            "high": 11.0,
            "low": 9.5,
            "close": 10.5,
            "adjusted_close": 10.4,
            "volume": 100,
        },
        {
            "date": "2016-01-02",
            "open": 11.0,
            "high": 12.0,
            "low": 10.5,
            "close": 11.5,
            "adjusted_close": 11.4,
            "volume": 200,
        },
    ]

    validity_start = pd.Timestamp("2016-01-01")
    validity_end = pd.Timestamp("2016-12-31")
    validity_window = (validity_start, validity_end)

    records: List[dict] = firm_utils.extract_ohlcv(
        ohlcv_dicts=ohlcv_dicts,
        current_symbol="ABC.US",
        current_validity_window=validity_window,
        current_cik="0000000001",
    )

    assert len(records) == 1
    row = records[0]
    assert row["ticker"] == "ABC.US"
    assert row["cik"] == "0000000001"
    assert row["trading_day"] == pd.Timestamp("2016-01-02")
    assert row["open"] == pytest.approx(11.0)
    assert row["adjusted_close"] == pytest.approx(11.4)
    assert row["volume"] == 200


def test_build_return_df_aggregates_ohlcv_for_active_firms(
    dummy_logger: DummyInfraLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Confirm that `build_return_df` iterates over active firms, applies ticker
    alias mappings, and aggregates OHLCV records returned by
    `extract_ohlcv`.

    Parameters
    ----------
    dummy_logger : DummyInfraLogger
        Logger stand-in whose methods are ignored.
    monkeypatch : pytest.MonkeyPatch
        Used to patch ticker alias mappings, HTTP layer, and the OHLCV helper.

    Returns
    -------
    None
        The test passes if the resulting DataFrame contains one row per
        returned OHLCV record with the aliased ticker.
    """
    validity_window_obj = SimpleNamespace(
        lower=pd.Timestamp("2016-01-01"),
        upper=pd.Timestamp("2017-01-01"),
    )

    active_firms = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "validity_window": [validity_window_obj],
            "cik": ["0000000001"],
        }
    )

    monkeypatch.setattr(firm_utils, "TICKER_ALIAS_MAPPING", {"ABC": "XYZ"})

    def fake_make_request(url: str) -> DummyResponseContext:
        """Return a list payload to satisfy the list-branch in build_return_df."""
        return DummyResponseContext(
            payload=[
                {
                    "date": "2016-01-02",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "adjusted_close": 10.4,
                    "volume": 100,
                }
            ]
        )

    def fake_extract_ohlcv(
        ohlcv_dicts: Sequence[Dict[str, Any]],
        current_symbol: str,
        current_validity_window: Any,
        current_cik: str,
    ) -> List[dict]:
        """Return a single mapped record per firm."""
        return [
            {
                "ticker": current_symbol,
                "cik": current_cik,
                "trading_day": pd.Timestamp("2016-01-02"),
                "open": 10.0,
                "high": 11.0,
                "low": 9.5,
                "close": 10.5,
                "adjusted_close": 10.4,
                "volume": 100,
            }
        ]

    monkeypatch.setattr(firm_utils, "make_request", fake_make_request)
    monkeypatch.setattr(firm_utils, "extract_ohlcv", fake_extract_ohlcv)

    logger = cast(InfraLogger, dummy_logger)
    df: pd.DataFrame = firm_utils.build_returns_df(
        active_firms=active_firms,
        api_key="TESTKEY",
        logger=logger,
        real_run=True,
    )

    assert df.shape[0] == 1
    assert df.loc[0, "ticker"] == "XYZ.US"
    assert df.loc[0, "trading_day"] == pd.Timestamp("2016-01-02")
    assert df.loc[0, "volume"] == 100


def test_align_fundamentals_with_returns_performs_backward_asof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure that `align_fundamentals_with_returns` performs a backward
    merge-as-of by ticker and date, mapping each trading day to the most
    recent filing date within the configured tolerance.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to patch `QUARTER_IN_DAYS` for deterministic tolerance.

    Returns
    -------
    None
        The test passes if trading days after the first filing date receive
        fundamentals from that filing date and pre-filing days receive NaNs.
    """
    monkeypatch.setattr(firm_utils, "QUARTER_IN_DAYS", 90)

    fundamentals_df = pd.DataFrame(
        {
            "ticker": ["ABC.US", "ABC.US"],
            "cik": ["0000000001", "0000000001"],
            "filing_date": [datetime(2016, 1, 10), datetime(2016, 4, 10)],
            "book_equity": [100.0, 200.0],
            "shares_outstanding": [10.0, 20.0],
        }
    )

    returns_df = pd.DataFrame(
        {
            "ticker": ["ABC.US", "ABC.US", "ABC.US"],
            "trading_day": [
                datetime(2015, 12, 31),
                datetime(2016, 1, 15),
                datetime(2016, 5, 1),
            ],
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.2, 11.2, 12.2],
            "adjusted_close": [10.2, 11.2, 12.2],
            "volume": [100, 200, 300],
        }
    )

    aligned: pd.DataFrame = firm_utils.align_fundamentals_with_returns(
        fundamentals_df=fundamentals_df.copy(),
        returns_df=returns_df.copy(),
    )

    # Three rows preserved.
    assert aligned.shape[0] == 3

    # Pre-filing trading day should have NaN fundamentals.
    first_row = aligned.iloc[0]
    assert pd.isna(first_row["book_equity"])
    assert pd.isna(first_row["shares_outstanding"])

    # Trading day between the two filings matches the first filing.
    middle_row = aligned.iloc[1]
    assert middle_row["book_equity"] == pytest.approx(100.0)
    assert middle_row["shares_outstanding"] == pytest.approx(10.0)

    # Trading day after the second filing matches the second filing.
    last_row = aligned.iloc[2]
    assert last_row["book_equity"] == pytest.approx(200.0)
    assert last_row["shares_outstanding"] == pytest.approx(20.0)


def test_calculate_features_computes_returns_and_controls_without_lookahead() -> None:
    """
    Confirm that `calculate_features` sorts by ticker and trading_day,
    computes market capitalization, log market cap, book-to-market, and
    return decompositions, and leaves early rows of realized volatility and
    momentum as NaN due to insufficient history.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the second row's returns match the expected log
        formulas based on the prior day's adjusted close and if realized
        volatilities and momentum signals do not use current-day information.
    """
    df = pd.DataFrame(
        {
            "ticker": ["ABC.US", "ABC.US", "ABC.US"],
            "trading_day": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.2, 11.2, 12.2],
            "adjusted_close": [10.0, 11.0, 12.0],
            "volume": [100, 200, 300],
            "book_equity": [1000.0, 1000.0, 1000.0],
            "shares_outstanding": [100.0, 100.0, 100.0],
        }
    )

    features: pd.DataFrame = firm_utils.calculate_features(df.copy())

    # Ensure rows are sorted by date.
    assert list(features["trading_day"]) == sorted(features["trading_day"])

    # Second row (index 1) should use previous adjusted_close = 10.0.
    second = features.iloc[1]
    prev_adj_close = 10.0
    expected_market_cap = second["adjusted_close"] * second["shares_outstanding"]
    assert second["market_cap"] == pytest.approx(expected_market_cap)
    assert second["log_market_cap"] == pytest.approx(np.log(expected_market_cap))
    assert second["book_to_market"] == pytest.approx(second["book_equity"] / expected_market_cap)

    expected_overnight = np.log(second["open"] / prev_adj_close)
    expected_intraday = np.log(second["adjusted_close"] / second["open"])
    expected_close_to_close = np.log(second["adjusted_close"] / prev_adj_close)

    assert second["overnight_log_return"] == pytest.approx(expected_overnight)
    assert second["intraday_log_return"] == pytest.approx(expected_intraday)
    assert second["close_to_close_log_return"] == pytest.approx(expected_close_to_close)

    # With only three observations, realized vol and momentum should be NaN
    # across all rows because the window sizes (21, 252) are not satisfied.
    assert features["realized_vol_21d"].isna().all()
    assert features["realized_vol_252d"].isna().all()
    assert features["momentum_1m"].isna().all()
    assert features["momentum_12m"].isna().all()
