"""
Purpose
-------
Database utilities and window-splitting helpers for data notebooks.
Provides a SQLAlchemy engine factory wired to environment variables and
a utility to split ticker–CIK validity windows around curated split dates.

Key behaviors
-------------
- Build a SQLAlchemy Engine for the Postgres warehouse using environment
  variables and a pinned UTC timezone option.
- Adjust evidence DataFrame validity_window values for tickers in
  WINDOW_SPLIT_PAIRS by:
  - splitting a long validity_window into pre- and post-split segments, and
  - dropping rows whose filed_at timestamp is inconsistent with the segment.

Conventions
-----------
- Database connection parameters are read from:
  POSTGRES_USER, POSTGRES_PASSWORD, DB_HOST, DB_PORT, POSTGRES_DB.
- The engine is created with `timezone=utc` to keep session behavior
  consistent across notebooks.
- evidence_df is expected to contain at least:
  'ticker', 'candidate_cik', 'validity_window', and 'filed_at'.
- validity_window is stored as a psycopg2.extras.DateRange with half-open
  bounds '[)' matching the research horizon and split rules.

Downstream usage
----------------
- Call connect_with_sqlalchemy() from notebooks to obtain a warehouse engine
  using the standard environment configuration.
- Call split_windows(evidence_df) before downstream adjudication/profile
  loading to ensure that special tickers in WINDOW_SPLIT_PAIRS have their
  validity windows split and rows filtered consistently with curated
  split dates.
"""

import os
from typing import Any, cast

import pandas as pd
import psycopg2.extras
import sqlalchemy as sa

from notebooks_utils.data_notebooks_utils.general_data_notebooks_config import WINDOW_SPLIT_PAIRS


def connect_with_sqlalchemy() -> sa.Engine:
    """
    Construct a SQLAlchemy Engine for the Postgres warehouse.

    Parameters
    ----------
    None

    Returns
    -------
    sqlalchemy.Engine
        A SQLAlchemy Engine instance connected to the Postgres database
        specified by the environment variables:
        - POSTGRES_USER
        - POSTGRES_PASSWORD
        - DB_HOST
        - DB_PORT
        - POSTGRES_DB
        with the session configured to use UTC timezone via
        `options=-c timezone=utc`.

    Raises
    ------
    KeyError
        If any of the required environment variables are missing.
    sqlalchemy.exc.SQLAlchemyError
        If SQLAlchemy fails to construct the engine for the given URL.

    Notes
    -----
    - This helper does not open any connections eagerly; it only returns
      an Engine object. Connections will be opened lazily when first used.
    - The connection URL is constructed using the standard
      `postgresql://user:pass@host:port/dbname` format with an `options`
      query parameter to force UTC.
    """

    db_user: str = os.environ["POSTGRES_USER"]
    db_password: str = os.environ["POSTGRES_PASSWORD"]
    db_host: str = os.environ["DB_HOST"]
    db_port: str = os.environ["DB_PORT"]
    db_name: str = os.environ["POSTGRES_DB"]
    options: str = "options=-c timezone=utc"
    database_url: str = (
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?{options}"
    )
    engine = sa.create_engine(database_url)
    return engine


def split_windows(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Split validity windows for configured tickers around curated split dates.

    Parameters
    ----------
    evidence_df : pandas.DataFrame
        Evidence-level DataFrame expected to contain, at minimum, the
        following columns:
        - 'ticker' : str
            Ticker symbol for the evidence row.
        - 'candidate_cik' : str
            Candidate CIK associated with the ticker.
        - 'validity_window' : object
            Existing psycopg2.extras.DateRange episode for the mapping.
        - 'filed_at' : pandas.Timestamp
            Filing timestamp used to decide whether a row belongs to the
            pre-split or post-split segment.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame where, for each WindowSplit in WINDOW_SPLIT_PAIRS:
        - Rows with matching (ticker, candidate_cik) are assigned a new
          validity_window:
            - pre_split_cik rows get [window.lower, split_date)
            - post_split_cik rows get [split_date, window.upper)
        - Rows whose filed_at is inconsistent with the split side
          (e.g., pre-split CIK but filed_at >= split_date) are removed.
        The index is reset after each split pass and in the final result.

    Raises
    ------
    KeyError
        If required columns ('ticker', 'candidate_cik', 'validity_window',
        'filed_at') are missing from evidence_df.
    AttributeError
        If any validity_window lacks lower/upper bounds compatible with
        the DateRange contract used by WINDOW_SPLIT_PAIRS.

    Notes
    -----
    - This function mutates and filters evidence_df within the loop, then
      returns the final filtered DataFrame. Callers should treat the result
      as the canonical version after splitting.
    - Only rows whose (ticker, candidate_cik) match a WindowSplit's
      pre_split_cik or post_split_cik are affected; all other rows pass
      through unchanged.
    - Split logic assumes that lower and upper bounds on each configured
      validity_window are finite (non-NULL) and that split_date lies within
      those bounds; these conditions are enforced by the configuration, not
      by runtime checks here.
    """

    for window_split in WINDOW_SPLIT_PAIRS:

        # Assertions for static analysis; Should always hold true given the
        # construction of WINDOW_SPLIT_PAIRS.
        lower_bound = window_split.validity_window.lower
        upper_bound = window_split.validity_window.upper
        assert lower_bound is not None
        assert upper_bound is not None

        identity_mask: pd.Series = (evidence_df["ticker"] == window_split.ticker) & (
            (evidence_df["candidate_cik"] == window_split.pre_split_cik)
            | (evidence_df["candidate_cik"] == window_split.post_split_cik)
        )
        pre_split_mask: pd.Series = identity_mask & (
            evidence_df["candidate_cik"] == window_split.pre_split_cik
        )
        post_split_mask: pd.Series = identity_mask & (
            evidence_df["candidate_cik"] == window_split.post_split_cik
        )
        evidence_df.loc[pre_split_mask, "validity_window"] = cast(
            Any,
            psycopg2.extras.DateRange(
                lower=lower_bound, upper=window_split.split_date.date(), bounds="[)"
            ),
        )
        evidence_df.loc[post_split_mask, "validity_window"] = cast(
            Any,
            psycopg2.extras.DateRange(
                lower=window_split.split_date.date(), upper=upper_bound, bounds="[)"
            ),
        )
        pre_split_keep_mask: pd.Series = pre_split_mask & (
            evidence_df["filed_at"] < window_split.split_date
        )
        post_split_keep_mask: pd.Series = post_split_mask & (
            evidence_df["filed_at"] >= window_split.split_date
        )
        remove_mask: pd.Series = ~(pre_split_keep_mask | post_split_keep_mask | ~identity_mask)
        evidence_df = evidence_df.loc[~remove_mask].reset_index(drop=True)
    return evidence_df
