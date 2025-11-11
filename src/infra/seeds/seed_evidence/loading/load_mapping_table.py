"""
Purpose
-------
Helpers for populating the ticker_cik_mapping table from a canonical
evidence DataFrame produced by the adjudication pipeline.

Key behaviors
-------------
- Generate the INSERT .. ON CONFLICT DO NOTHING statement used to
  load mapping episodes.
- Convert a canonical evidence DataFrame into a row generator of
  tuples suitable for batch loading via load_into_table.
- Optionally execute the load into Postgres, wiring DB credentials
  from environment variables with python-dotenv.

Conventions
-----------
- Input frames are expected to contain, at minimum, the columns:
  'ticker', 'candidate_cik', 'validity_window', 'form_type',
  'filed_at', 'source', 'accession_num', and 'evidence_id'.
- Database connectivity is delegated to infra.utils.db_utils:
  connect_to_db() returns a context-managed connection and
  load_into_table() performs the actual batch insert.
- The target table schema is ticker_cik_mapping, with episodes
  modeled as half-open validity windows.

Downstream usage
----------------
- Use load_mapping_table(...) at the end of the adjudication /
  canonicalization pipeline to materialize accepted ticker→CIK
  mapping episodes into the warehouse.
- Use generate_mapping_query() and create_mapping_row_generator()
  directly when you need finer control over connection lifecycle
  or batching strategy.
"""

from typing import Iterator

import pandas as pd
from dotenv import load_dotenv

from infra.utils.db_utils import connect_to_db, load_into_table


def load_mapping_table(canonical_evidence_df: pd.DataFrame, load: bool = False) -> None:
    """
    Optionally load canonical ticker–CIK mappings into ticker_cik_mapping.

    Parameters
    ----------
    canonical_evidence_df : pandas.DataFrame
        DataFrame containing one row per canonical evidence record for a
        ticker–CIK episode. Must include at least:
        - 'ticker'
        - 'candidate_cik'
        - 'validity_window'
        - 'form_type'
        - 'filed_at'
        - 'source'
        - 'accession_num'
        - 'evidence_id'
    load : bool, default False
        Guard flag; when False the function returns immediately without
        touching the database. When True, the function will:
        - load environment variables via python-dotenv,
        - open a DB connection via connect_to_db(), and
        - call load_into_table(...) with a row generator and INSERT query.

    Returns
    -------
    None
        This function is executed for its side effect of inserting rows
        into ticker_cik_mapping when load=True.

    Raises
    ------
    KeyError
        If required columns are missing from canonical_evidence_df.
    psycopg2.Error
        If database connectivity or insertion fails inside connect_to_db
        or load_into_table.

    Notes
    -----
    - The load flag is intended to make this helper safe to call from
      notebooks: you can pass load=False while iterating on logic and
      flip it to True only when you are ready to persist mappings.
    - The INSERT statement uses `ON CONFLICT DO NOTHING`, so re-running
      the load with the same input is idempotent with respect to the
      primary key on ticker_cik_mapping.
    """

    if not load:
        return None

    load_dotenv()
    query: str = generate_mapping_query()
    generator: Iterator[tuple] = create_mapping_row_generator(canonical_evidence_df)
    with connect_to_db() as conn:
        load_into_table(conn, generator, query)
    return None


def generate_mapping_query() -> str:
    """
    Build the INSERT statement used to populate ticker_cik_mapping.

    Parameters
    ----------
    None

    Returns
    -------
    str
        A parametrized SQL string of the form:
        INSERT INTO ticker_cik_mapping (...)
        VALUES %s
        ON CONFLICT DO NOTHING;
        intended for use with a batch loader such as psycopg2.extras.execute_values
        inside load_into_table.

    Raises
    ------
    None

    Notes
    -----
    - The VALUES placeholder is a single `%s`, assuming that the caller
      will pass a sequence of tuples for efficient bulk insertion.
    - ON CONFLICT DO NOTHING ensures duplicate episodes (by PK) are
      ignored rather than causing the batch to fail.
    """

    return """
    INSERT INTO ticker_cik_mapping (
        ticker,
        cik,
        validity_window,
        evidence_type,
        filed_at,
        source,
        accession_num,
        evidence_id
    ) VALUES %s
    ON CONFLICT DO NOTHING;
    """


def create_mapping_row_generator(
    canonical_evidence_df: pd.DataFrame,
) -> Iterator[tuple]:
    """
    Yield ticker_cik_mapping rows from a canonical evidence DataFrame.

    Parameters
    ----------
    canonical_evidence_df : pandas.DataFrame
        DataFrame containing canonical evidence for ticker–CIK episodes.
        Must include the columns:
        - 'ticker'
        - 'candidate_cik'
        - 'validity_window'
        - 'form_type'
        - 'filed_at'
        - 'source'
        - 'accession_num'
        - 'evidence_id'

    Returns
    -------
    Iterator[tuple]
        An iterator of tuples, one per row in canonical_evidence_df, in
        the order expected by ticker_cik_mapping:
        (ticker, cik, validity_window, evidence_type, filed_at,
         source, accession_num, evidence_id).

    Raises
    ------
    KeyError
        If any of the required columns are missing from
        canonical_evidence_df.

    Notes
    -----
    - This generator is intended to be consumed by load_into_table.
    - The function uses DataFrame.iterrows(), favoring clarity over
      maximum throughput; if performance becomes an issue, a vectorized
      or itertuples-based implementation can be substituted without
      changing the yielded tuple shape.
    """

    for _, row in canonical_evidence_df.iterrows():
        yield (
            row["ticker"],
            row["candidate_cik"],
            row["validity_window"],
            row["form_type"],
            row["filed_at"],
            row["source"],
            row["accession_num"],
            row["evidence_id"],
        )
