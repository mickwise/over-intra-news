"""
Purpose
-------
Load the firm-level `features_df` DataFrame into the
`equity_regression_panel` table used for the regressions.

Key behaviors
-------------
- Generate a parameterized INSERT statement compatible with
  `psycopg2.extras.execute_values` via `load_into_table`.
- Convert each row of `features_df` into a tuple matching the DDL column
  order of `equity_regression_panel`.
- Support a dry-run mode via the `real_run` flag so notebooks can execute
  end-to-end without writing to the database by default.

Conventions
-----------
- `features_df` is assumed to contain one row per (cik, trading_day) with
  columns derived from `firm_regressors_utils.calculate_features`.
- `ticker` in `features_df` is treated as the vendor symbol and loaded
  into `eodhd_symbol` in the target table.
- All returns are log returns and all rolling controls (realized
  volatility, momentum) are computed without look-ahead.

Downstream usage
----------------
Call `load_equity_regression_panel(features_df, real_run=True)` from a
data notebook once `features_df` has been fully constructed and
validated. Use `real_run=False` during development to avoid mutating the
database.
"""

from typing import Iterator

import pandas as pd

from infra.utils.db_utils import connect_to_db, load_into_table


def load_equity_regression_panel(
    features_df: pd.DataFrame,
    real_run: bool = False,
) -> None:
    """
    Load a firm-level features DataFrame into `equity_regression_panel`.

    Parameters
    ----------
    features_df : pandas.DataFrame
        DataFrame produced by `calculate_features`, containing one row per
        (cik, trading_day) with price-based returns, volatility controls,
        and fundamentals-based controls.
    real_run : bool, optional
        If False (default), perform no database writes. If True, execute
        the INSERT statement against the configured database.

    Returns
    -------
    None
        The function returns None whether or not data is written.

    Notes
    -----
    - When `real_run` is True, rows are inserted using the query returned
      by `generate_equity_regression_panel_query` and the tuples emitted
      by `create_equity_regression_panel_row_generator`.
    - Duplicate (cik, trading_day) rows are ignored by the ON CONFLICT
      clause in the INSERT.
    """
    if not real_run:
        return None

    equity_regression_panel_query: str = generate_equity_regression_panel_query()
    row_generator: Iterator[tuple] = create_equity_regression_panel_row_generator(features_df)

    with connect_to_db() as conn:
        load_into_table(
            conn,
            row_generator,
            equity_regression_panel_query,
        )


def generate_equity_regression_panel_query() -> str:
    """
    Build the INSERT statement for loading `equity_regression_panel`.

    Parameters
    ----------
    None

    Returns
    -------
    str
        A multi-line SQL string suitable for use with
        `psycopg2.extras.execute_values`, where the VALUES placeholder is
        `%s` and each record is a tuple matching the column list.

    Notes
    -----
    - The column order matches the tuple emitted by
      `create_equity_regression_panel_row_generator`.
    - The ON CONFLICT clause targets the primary key (cik, trading_day)
      and performs a no-op for duplicates.
    """
    return """
    INSERT INTO equity_regression_panel (
        cik,
        trading_day,
        eodhd_symbol,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
        adjusted_close_price,
        overnight_log_return,
        intraday_log_return,
        close_to_close_log_return,
        realized_vol_21d,
        realized_vol_252d,
        momentum_1m,
        momentum_12m,
        shares_outstanding,
        market_cap,
        log_market_cap,
        book_to_market,
        filing_date
    ) VALUES %s
    ON CONFLICT DO NOTHING;
    """


def create_equity_regression_panel_row_generator(
    features_df: pd.DataFrame,
) -> Iterator[tuple]:
    """
    Yield row tuples for insertion into `equity_regression_panel`.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Features DataFrame with at least the columns:
        ['cik', 'trading_day', 'ticker', 'open', 'high', 'low', 'close',
         'volume', 'adjusted_close', 'overnight_log_return',
         'intraday_log_return', 'close_to_close_log_return',
         'realized_vol_21d', 'realized_vol_252d', 'momentum_1m',
         'momentum_12m', 'shares_outstanding', 'market_cap',
         'log_market_cap', 'book_to_market', 'filing_date'].

    Returns
    -------
    Iterator[tuple]
        An iterator of tuples whose positional order matches the column
        order in `generate_equity_regression_panel_query`.

    Notes
    -----
    - `ticker` is treated as the EODHD vendor symbol and mapped into the
      `eodhd_symbol` column of the target table.
    - `trading_day` is expected to be timezone-naive and compatible with
      the DATE column in PostgreSQL.
    """
    for _, row in features_df.iterrows():
        yield (
            row["cik"],
            row["trading_day"],
            row["ticker"],  # eodhd_symbol
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"],
            row["adjusted_close"],
            row["overnight_log_return"],
            row["intraday_log_return"],
            row["close_to_close_log_return"],
            row["realized_vol_21d"],
            row["realized_vol_252d"],
            row["momentum_1m"],
            row["momentum_12m"],
            row["shares_outstanding"] if not pd.isna(row["shares_outstanding"]) else None,
            row["market_cap"] if not pd.isna(row["market_cap"]) else None,
            row["log_market_cap"] if not pd.isna(row["log_market_cap"]) else None,
            row["book_to_market"] if not pd.isna(row["book_to_market"]) else None,
            row["filing_date"] if not pd.isna(row["filing_date"]) else None,
        )
