"""
Purpose
-------
Utilities for normalizing company names and loading auto-accepted profile
episodes into `security_profile_history`.

Key behaviors
-------------
- Canonicalize raw company-name strings into a normalized key used for grouping
  and matching.
- Stage auto-accepted ticker/CIK episodes into a temporary table and insert
  corresponding profile rows into `security_profile_history`.

Conventions
-----------
- Company-name canonicalization is upper-case, punctuation-stripped, and
  suffix-stripped based on `NAME_SUFFIXES_TO_REMOVE`, with all spaces removed.
- Auto-accepted names are staged via `AUTO_PICK_TABLE` and then written into
  `security_profile_history` via a single INSERT .. SELECT statement.
- `security_profile_history` episodes use half-open `[start, end)` date ranges
  and are de-duplicated by `(cik, validity_window)` at the database level.

Downstream usage
----------------
- Use `company_name_canonicalizer` to build canonical name keys for clustering
  or matching candidate company names.
- Call `load_auto_accepted_names` in the notebook/ETL pipeline to materialize
  auto-accepted profile episodes in `security_profile_history` once an engine
  and an `auto_pick_names_df` are available.
"""

import pandas as pd
import sqlalchemy as sa

# fmt: off
from notebooks_utils.data_notebooks_utils.security_master_profiles_membership.\
  security_master_profiles_memberships_config import (
    AUTO_PICK_TABLE,
    NAME_SUFFIXES_TO_REMOVE,
)

# fmt: on


def company_name_canonicalizer(company_names: pd.Series) -> pd.Series:
    """
    Canonicalize company-name strings for matching and clustering.

    Parameters
    ----------
    company_names : pandas.Series
        Series of raw company-name strings (any casing, punctuation, and
        spacing). May contain nulls; these are propagated through the
        transformations.

    Returns
    -------
    pandas.Series
        Series of canonicalized company-name keys where:
        - names are uppercased and stripped of leading/trailing whitespace,
        - punctuation characters `[,&-/ .]` are removed,
        - configured suffixes in `NAME_SUFFIXES_TO_REMOVE` are removed as
          whole words (optionally followed by a period),
        - all remaining spaces are removed.

    Notes
    -----
    - This function is deliberately lossy: its goal is to collapse spelling /
      formatting variants of the same legal name into a single key.
    - Suffix removal is driven by `NAME_SUFFIXES_TO_REMOVE` from the
      configuration module.
    """

    canonicalized_company_names: pd.Series = company_names.str.upper().str.strip()
    canonicalized_company_names = canonicalized_company_names.str.replace(
        r"[,&\-\/.]", "", regex=True
    )
    for suffix in NAME_SUFFIXES_TO_REMOVE:
        canonicalized_company_names = canonicalized_company_names.str.replace(
            rf"\b{suffix}\b\.?", "", regex=True
        )
    return canonicalized_company_names.str.replace(" ", "")


def load_auto_accepted_names(
    auto_pick_names_df: pd.DataFrame, engine: sa.Engine | None = None, load_table: bool = False
) -> None:
    """
    Load auto-accepted profile episodes into `security_profile_history`.

    Parameters
    ----------
    auto_pick_names_df : pandas.DataFrame
        DataFrame of auto-accepted candidate mappings, expected to contain at
        least the columns:
        - 'ticker'
        - 'validity_window' (as text or range-like representation)
        - 'candidate_cik'
    engine : sqlalchemy.Engine or None, optional
        SQLAlchemy engine used to connect to the warehouse. If None, no
        database work is attempted regardless of `load_table`.
    load_table : bool, optional
        When True and an engine is provided, this function will:
        1) Stage the distinct `(ticker, validity_window, candidate_cik)` rows
           into a temporary table named `AUTO_PICK_TABLE`.
        2) Execute an INSERT .. SELECT into `security_profile_history` that
           joins `AUTO_PICK_TABLE`, `ticker_cik_mapping`, and
           `ticker_cik_evidence`.
        3) Drop the staging table.

    Returns
    -------
    None
        This function operates via database side effects only and does not
        return a value.

    Notes
    -----
    - The INSERT uses `SELECT DISTINCT ON (cik, validity_window)` to pick one
      representative evidence row per `(cik, validity_window)` before inserting
      into `security_profile_history`.
    - The INSERT statement is written with `ON CONFLICT (cik, validity_window)
      DO NOTHING` to make the operation idempotent.
    - If `load_table` is False or `engine` is None, the function is a no-op.
    """

    if load_table and engine is not None:
        security_profile_history_ingestion_query = f"""
        INSERT INTO security_profile_history (
            cik,
            validity_window,
            company_name,
            source,
            evidence_id
        )
        SELECT DISTINCT ON (mapping_table.cik, mapping_table.validity_window)
            mapping_table.cik,
            mapping_table.validity_window,
            evidence_table.company_name,
            evidence_table.source,
            mapping_table.evidence_id
        FROM {AUTO_PICK_TABLE} AS auto_pick_table
        JOIN ticker_cik_mapping AS mapping_table
        ON mapping_table.ticker = auto_pick_table.ticker
        AND mapping_table.cik    = auto_pick_table.cik
        AND mapping_table.validity_window = auto_pick_table.validity_window::daterange
        JOIN ticker_cik_evidence AS evidence_table
        ON evidence_table.evidence_id = mapping_table.evidence_id
        ORDER BY
            mapping_table.cik,
            mapping_table.validity_window,
            mapping_table.evidence_id
        ON CONFLICT (cik, validity_window) DO NOTHING;
        """
        auto_pick_staging = (
            auto_pick_names_df[["ticker", "validity_window", "candidate_cik"]]
            .drop_duplicates()
            .rename(columns={"candidate_cik": "cik"})
        )
        auto_pick_staging.to_sql(
            AUTO_PICK_TABLE,
            engine,
            if_exists="replace",
            index=False,
        )
        with engine.begin() as conn:
            conn.execute(sa.text(security_profile_history_ingestion_query))
            conn.execute(sa.text(f"DROP TABLE {AUTO_PICK_TABLE};"))
