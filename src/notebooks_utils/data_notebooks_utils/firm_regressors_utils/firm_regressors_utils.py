"""
Purpose
-------
Build the firm-level regression feature panel used in the Glasserman-style
news regressions by fetching fundamentals and OHLCV data, aligning them
as-of each trading day, and computing returns and control variables.

Key behaviors
-------------
- Extract the set of "active" firms with news coverage from the
  `ticker_cik_mapping` table.
- Pull quarterly fundamentals and daily OHLCV data from EODHD for each
  active firm, filtered by its validity window.
- Align fundamentals to trading days via a backward merge-as-of.
- Construct market-cap, book-to-market, log-return decompositions,
  realized volatilities, and momentum signals with strict no-look-ahead.

Conventions
-----------
- Tickers are converted to EODHD symbols (e.g., `ABC` → `ABC.US` or an
  alias from `TICKER_ALIAS_MAPPING`) before hitting vendor APIs.
- Validity windows are treated as half-open intervals `[start, end)` and
  used to filter both fundamentals and OHLCV records.
- All returns are computed as natural-log returns; realized vol is
  annualized with a 252-day trading-year convention.
- Network calls and database connections are injected via helpers and
  can be monkeypatched in tests.

Downstream usage
----------------
Typical usage is:
1. Call `extract_active_firms` to identify the relevant ticker–CIK
   episodes.
2. Call `build_fundamentals_df` and `build_return_df` (with `real_run=True`)
   to materialize fundamentals and OHLCV DataFrames.
3. Call `align_fundamentals_with_returns` followed by
   `calculate_features` to obtain the final `features_df` used for
   regressions and loading into `equity_regression_panel`.
"""

from typing import Any, List

import numpy as np
import pandas as pd
import sqlalchemy as sa

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindow
from infra.utils.requests_utils import make_request
from notebooks_utils.data_notebooks_utils.firm_regressors_utils.firm_regressors_config import (
    FUNDAMENTALS_FILTER,
    QUARTER_IN_DAYS,
    TICKER_ALIAS_MAPPING,
)
from notebooks_utils.data_notebooks_utils.general_data_notebooks_utils import (
    connect_with_sqlalchemy,
)


def extract_active_firms() -> pd.DataFrame:
    """
    Extract the set of ticker–CIK episodes that actually appear in the news
    corpus and should enter the regression universe.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
        - 'ticker' : str
            Canonical exchange ticker from `ticker_cik_mapping`.
        - 'validity_window' : psycopg2 date range
            Half-open `[start, end)` window over which this ticker→CIK
            episode is valid.
        - 'cik' : str
            Zero-padded 10-digit SEC CIK for the firm.

    Notes
    -----
    - The SQL query filters to ticker–CIK episodes that appear in
      `parsed_news_articles` (via `lda_documents`) so that only firms with
      news exposure are carried forward.
    - A SQLAlchemy engine is obtained via `connect_with_sqlalchemy` and
      passed into `pandas.read_sql`.
    """

    active_firms_query: str = """
        SELECT
            m.ticker,
            m.validity_window,
            m.cik
        FROM ticker_cik_mapping AS m
        WHERE EXISTS (
            SELECT 1
            FROM parsed_news_articles AS pna
            JOIN lda_documents AS ld
              ON ld.article_id = pna.article_id
            WHERE m.cik = ANY (pna.cik_list)
        );
    """
    engine: sa.Engine = connect_with_sqlalchemy()
    return pd.read_sql(active_firms_query, engine)


def build_fundamentals_df(
    active_firms: pd.DataFrame, api_key: str, logger: InfraLogger, real_run: bool = False
) -> pd.DataFrame:
    """
    Fetch quarterly fundamentals for each active firm from EODHD and map them
    into a flat per-filing DataFrame.

    Parameters
    ----------
    active_firms : pandas.DataFrame
        Output of `extract_active_firms`, containing ticker, validity_window,
        and cik for each ticker→CIK episode.
    api_key : str
        EODHD API token used to authenticate the fundamentals requests.
    logger : InfraLogger
        Logger used for debug and warning messages during ingestion.
    real_run : bool, optional
        When False, return an empty DataFrame without making any HTTP calls.
        Intended for dry runs and tests. Default is False.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per accepted quarterly fundamentals record
        and columns:
        - 'ticker' : str
            EODHD symbol (e.g., 'ABC.US') after alias mapping.
        - 'cik' : str
            SEC CIK inherited from `active_firms`.
        - 'filing_date' : pandas.Timestamp
            Normalized filing date of the quarterly report.
        - 'book_equity' : float or None
            Total assets minus total liabilities when both are present.
        - 'shares_outstanding' : float or None
            Common shares outstanding, if reported.

    Notes
    -----
    - Fundamentals are skipped if:
    - the JSON payload is not a dict, or
    - the entry is older than `QUARTER_IN_DAYS` before the validity-window
        start, or
    - `filing_date` is missing.
    - All HTTP calls are made through `make_request`, and URLs are built via
    `construct_fundamentals_url`.
    """

    if not real_run:
        return pd.DataFrame()
    records: List[dict] = []
    for _, row in active_firms.iterrows():
        current_symbol: str
        if row["ticker"] in TICKER_ALIAS_MAPPING:
            current_symbol = TICKER_ALIAS_MAPPING[row["ticker"]] + ".US"
        else:
            current_symbol = row["ticker"] + ".US"
        current_validity_window: ValidityWindow = (
            pd.Timestamp(row["validity_window"].lower),
            pd.Timestamp(row["validity_window"].upper),
        )
        current_cik: str = row["cik"]
        logger.debug(
            event="build_fundamentals_df", msg=f"Fetching fundamentals for {current_symbol}"
        )
        url: str = construct_fundamentals_url(current_symbol, api_key)
        with make_request(url) as response:
            quarterly_dict: Any = response.json()
            if not isinstance(quarterly_dict, dict):
                logger.warning(
                    event="build_fundamentals_df",
                    msg=(
                        f"Fundamentals data for {current_symbol} is not a dict. "
                        f"Response: {quarterly_dict}"
                    ),
                )
                continue
            current_records: List[dict] = extract_quarterly_data(
                quarterly_dict, current_symbol, current_validity_window, current_cik, logger
            )
            records.extend(current_records)
        logger.debug(
            event="build_fundamentals_df",
            msg=f"Completed fetching fundamentals for {current_symbol}",
        )
    return pd.DataFrame.from_records(records)


def construct_fundamentals_url(symbol: str, api_key: str) -> str:
    """
    Build the EODHD fundamentals endpoint URL for a given symbol and API key.

    Parameters
    ----------
    symbol : str
        EODHD symbol to query (e.g., 'ABC.US').
    api_key : str
        EODHD API token to authenticate the request.

    Returns
    -------
    str
        Fully formatted fundamentals URL including the configured
        `FUNDAMENTALS_FILTER`, the API token, and `fmt=json`.

    Notes
    -----
    - The filter string is taken from `FUNDAMENTALS_FILTER` so tests can
      monkeypatch it without duplicating literals.
    """

    return (
        f"https://eodhd.com/api/fundamentals/{symbol}?"
        f"filter={FUNDAMENTALS_FILTER}&api_token={api_key}&fmt=json"
    )


def extract_quarterly_data(
    quarterly_dict: dict[str, dict],
    current_symbol: str,
    current_validity_window: ValidityWindow,
    current_cik: str,
    logger: InfraLogger,
) -> List[dict]:
    """
    Normalize a raw fundamentals JSON dict into a list of quarterly records
    within the firm's validity window.

    Parameters
    ----------
    quarterly_dict : dict[str, str]
        Mapping from reporting date (e.g., '2016-06-30') to a dict of
        fundamentals fields as returned by EODHD.
    current_symbol : str
        EODHD symbol for the firm (e.g., 'ABC.US').
    current_validity_window : ValidityWindow
        Half-open `[start, end)` window during which the ticker→CIK episode
        is valid.
    current_cik : str
        SEC CIK for the firm associated with the current ticker episode.
    logger : InfraLogger
        Logger used to emit warnings when records are skipped.

    Returns
    -------
    list[dict]
        A list of dictionaries, each with keys:
        - 'ticker', 'cik', 'filing_date', 'book_equity',
        'shares_outstanding'.

    Notes
    -----
    - Records whose reporting date is more than `QUARTER_IN_DAYS` before the
      validity-window start are dropped.
    - Records with missing `filing_date` are skipped with a warning.
    - `book_equity` is computed as totalAssets − totalLiab when both fields
      are present; otherwise it is set to None.
    """
    current_records: List[dict] = []
    for key, value_dict in quarterly_dict.items():
        if pd.Timestamp(key) < current_validity_window[0] - pd.Timedelta(days=QUARTER_IN_DAYS):
            continue
        raw_current_filing_date: str | None = value_dict.get("filing_date")
        if raw_current_filing_date is None:
            logger.warning(
                event="extract_quarterly_data",
                msg=(
                    f"Filing date is missing for {current_symbol} in fundamentals data. "
                    f"Skipping this record."
                ),
                context={"date": key},
            )
            continue
        current_filing_date: pd.Timestamp = pd.Timestamp(raw_current_filing_date)
        within_validity_window: bool = (
            current_filing_date >= current_validity_window[0]
            and current_filing_date < current_validity_window[1]
        )
        if within_validity_window:
            current_total_assets: str | None = value_dict.get("totalAssets")
            current_liabilities: str | None = value_dict.get("totalLiab")
            total_asset_liabilities_missing: bool = (
                current_total_assets is None or current_liabilities is None
            )
            current_records.append(
                {
                    "ticker": current_symbol,
                    "cik": current_cik,
                    "filing_date": current_filing_date,
                    "book_equity": (
                        (float(current_total_assets) - float(current_liabilities))  # type: ignore
                        if not total_asset_liabilities_missing
                        else None
                    ),
                    "shares_outstanding": (
                        float(value_dict["commonStockSharesOutstanding"])
                        if value_dict["commonStockSharesOutstanding"] is not None
                        else None
                    ),
                }
            )
    return current_records


def build_returns_df(
    active_firms: pd.DataFrame, api_key: str, logger: InfraLogger, real_run: bool = False
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for each active firm from EODHD and assemble a
    per-firm trading-day panel that carries the firm's CIK.

    Parameters
    ----------
    active_firms : pandas.DataFrame
        Output of `extract_active_firms`, containing ticker, validity_window,
        and cik for each ticker→CIK episode.
    api_key : str
        EODHD API token used to authenticate the OHLCV requests.
    logger : InfraLogger
        Logger used for debug and warning messages during ingestion.
    real_run : bool, optional
        When False, return an empty DataFrame without making any HTTP calls.
        Intended for dry runs and tests. Default is False.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per accepted trading day and columns:
        - 'ticker' : str
            EODHD symbol (e.g., 'ABC.US') after alias mapping.
        - 'trading_day' : pandas.Timestamp
            Trading date within the ticker's validity window.
        - 'open', 'high', 'low', 'close', 'adjusted_close' : float
            Daily OHLC prices as returned by EODHD.
        - 'volume' : int
            Daily trading volume (shares).
        - 'cik' : str
            SEC CIK propagated from `active_firms` for every trading day.

    Notes
    -----
    - OHLCV payloads that are not lists are skipped with a warning.
    - Date filtering uses the half-open validity window `[start, end)` to
      drop out-of-range trading days.
    """

    if not real_run:
        return pd.DataFrame()
    records: List[dict] = []
    for _, row in active_firms.iterrows():
        current_cik: str = row["cik"]
        current_symbol: str
        if row["ticker"] in TICKER_ALIAS_MAPPING:
            current_symbol = TICKER_ALIAS_MAPPING[row["ticker"]] + ".US"
        else:
            current_symbol = row["ticker"] + ".US"
        current_validity_window: ValidityWindow = (
            pd.Timestamp(row["validity_window"].lower),
            pd.Timestamp(row["validity_window"].upper),
        )
        url: str = (
            f"https://eodhd.com/api/eod/{current_symbol}?period=d&api_token={api_key}&fmt=json"
        )
        with make_request(url) as response:
            ohlcv_dicts: Any = response.json()
            if not isinstance(ohlcv_dicts, list):
                logger.warning(
                    event="build_return_df",
                    msg=(
                        f"OHLCV data for {current_symbol} is not a list. "
                        f"Response: {ohlcv_dicts}"
                    ),
                )
                continue
            current_records: List[dict] = extract_ohlcv(
                ohlcv_dicts, current_symbol, current_validity_window, current_cik
            )
            records.extend(current_records)
    return pd.DataFrame.from_records(records)


def extract_ohlcv(
    ohlcv_dicts: List[dict],
    current_symbol: str,
    current_validity_window: ValidityWindow,
    current_cik: str,
) -> List[dict]:
    """
    Filter raw OHLCV payloads by validity window and map them into a flat
    list of per-day records.

    Parameters
    ----------
    ohlcv_dicts : list[dict]
        List of daily OHLCV dictionaries from EODHD, each containing at
        least 'date', 'open', 'high', 'low', 'close', 'adjusted_close',
        and 'volume'.
    current_symbol : str
        EODHD symbol for the firm (e.g., 'ABC.US').
    current_validity_window : ValidityWindow
        Half-open `[start, end)` window during which the ticker→CIK episode
        is valid.

    Returns
    -------
    list[dict]
        A list of dictionaries, each with keys:
        - 'ticker', 'trading_day', 'open', 'high', 'low',
          'close', 'adjusted_close', 'volume'.

    Notes
    -----
    - Only records whose `date` lies inside the validity window are
      retained.
    """

    current_records: List[dict] = []
    for record in ohlcv_dicts:
        current_date: pd.Timestamp = pd.Timestamp(record["date"])
        if current_date >= current_validity_window[0] and current_date < current_validity_window[1]:
            current_records.append(
                {
                    "ticker": current_symbol,
                    "trading_day": current_date,
                    "cik": current_cik,
                    "open": record["open"],
                    "high": record["high"],
                    "low": record["low"],
                    "close": record["close"],
                    "adjusted_close": record["adjusted_close"],
                    "volume": record["volume"],
                }
            )
    return current_records


def align_fundamentals_with_returns(
    fundamentals_df: pd.DataFrame, returns_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Align quarterly fundamentals with daily OHLCV returns using a backward
    merge-as-of.

    Parameters
    ----------
    fundamentals_df : pandas.DataFrame
        Output of `build_fundamentals_df`, containing at least 'ticker',
        'cik', 'filing_date', 'book_equity', and 'shares_outstanding'.
    returns_df : pandas.DataFrame
        Output of `build_return_df`, containing at least 'ticker',
        'trading_day', OHLCV columns, and a non-null 'cik' per row.

    Returns
    -------
    pandas.DataFrame
        A DataFrame (`features_df`) with one row per trading day and
        columns from both inputs, including:
        - 'ticker', 'trading_day', OHLCV fields, 'cik',
        - 'filing_date', 'book_equity', 'shares_outstanding'.

    Notes
    -----
    - Dates are normalized to midnight and sorted before performing
      `pandas.merge_asof` with `direction='backward'` and a tolerance of
      `QUARTER_IN_DAYS`.
    - The CIK column in fundamentals_df that was added for auditing and
      purposes is dropped before the merge.
    """

    fundamentals_df["filing_date"] = pd.to_datetime(fundamentals_df["filing_date"]).dt.normalize()
    returns_df["trading_day"] = pd.to_datetime(returns_df["trading_day"]).dt.normalize()
    fundamentals_df.sort_values(by=["filing_date"], inplace=True)
    returns_df.sort_values(by=["trading_day"], inplace=True)
    tolerance = pd.Timedelta(days=QUARTER_IN_DAYS)
    fundamentals_df = fundamentals_df.drop(columns=["cik"], errors="ignore")
    features_df: pd.DataFrame = pd.merge_asof(
        left=returns_df,
        right=fundamentals_df,
        left_on="trading_day",
        right_on="filing_date",
        by="ticker",
        direction="backward",
        tolerance=tolerance,
    )
    return features_df


def calculate_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived return and control features from the aligned
    fundamentals–returns panel without introducing look-ahead.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Output of `align_fundamentals_with_returns`, containing at least:
        'ticker', 'trading_day', 'open', 'adjusted_close', 'book_equity',
        'shares_outstanding', and 'cik'.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame sorted by 'ticker' and 'trading_day', augmented
        with:
        - 'market_cap' : float
            Price × shares_outstanding using adjusted close.
        - 'log_market_cap' : float
            Natural log of market_cap.
        - 'book_to_market' : float
            book_equity / market_cap when both are present.
        - 'overnight_log_return' : float
            log(open_t / adjusted_close_{t-1}).
        - 'intraday_log_return' : float
            log(adjusted_close_t / open_t).
        - 'close_to_close_log_return' : float
            log(adjusted_close_t / adjusted_close_{t-1}).
        - 'realized_vol_21d', 'realized_vol_252d' : float
            Annualized realized vol based on 21- and 252-day rolling
            standard deviations of close-to-close log returns, shifted by
            one day to avoid look-ahead.
        - 'momentum_1m', 'momentum_12m' : float
            1-month and 12-month percentage-return momentum signals,
            computed from adjusted-close levels and shifted by one day.

    Notes
    -----
    - All group-wise rolling operations are performed by ticker and then
      shifted so that only information strictly prior to `trading_day` is
      used.
    - Early rows that lack sufficient history for a given window naturally
      receive NaN in the corresponding realized-vol or momentum columns.
    """

    features_df.sort_values(by=["ticker", "trading_day"], inplace=True)
    features_df["market_cap"] = features_df["adjusted_close"] * features_df["shares_outstanding"]
    features_df["log_market_cap"] = np.log(features_df["market_cap"])
    features_df["book_to_market"] = features_df["book_equity"] / features_df["market_cap"]
    prev_adj_close = features_df.groupby("ticker")["adjusted_close"].shift(1)
    features_df["overnight_log_return"] = np.log(features_df["open"] / prev_adj_close)
    features_df["intraday_log_return"] = np.log(features_df["adjusted_close"] / features_df["open"])
    features_df["close_to_close_log_return"] = np.log(
        features_df["adjusted_close"] / prev_adj_close
    )
    features_df["realized_vol_21d"] = features_df.groupby("ticker")[
        "close_to_close_log_return"
    ].rolling(window=21).std().reset_index(level=0, drop=True) * np.sqrt(252)
    features_df["realized_vol_21d"] = features_df.groupby("ticker")["realized_vol_21d"].shift(1)
    features_df["realized_vol_252d"] = features_df.groupby("ticker")[
        "close_to_close_log_return"
    ].rolling(window=252).std().reset_index(level=0, drop=True) * np.sqrt(252)
    features_df["realized_vol_252d"] = features_df.groupby("ticker")["realized_vol_252d"].shift(1)
    features_df["momentum_1m"] = features_df.groupby("ticker")["adjusted_close"].pct_change(
        periods=21, fill_method=None
    )
    features_df["momentum_1m"] = features_df.groupby("ticker")["momentum_1m"].shift(1)
    features_df["momentum_12m"] = features_df.groupby("ticker")["adjusted_close"].pct_change(
        periods=252, fill_method=None
    )
    features_df["momentum_12m"] = features_df.groupby("ticker")["momentum_12m"].shift(1)
    return features_df
