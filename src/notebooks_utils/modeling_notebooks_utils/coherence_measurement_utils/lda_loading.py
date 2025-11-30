"""
Purpose
-------
Provide thin helpers for bulk-loading LDA outputs into the relational store,
bridging pandas DataFrames produced by the coherence notebook and the Postgres
tables that back downstream analysis.

Key behaviors
-------------
- Generate parametrized INSERT statements for `lda_article_topic_exposure`
  and `lda_topic_metadata` with idempotent ON CONFLICT handling.
- Convert topic exposure and topic metadata DataFrames into row iterables
  suitable for efficient bulk COPY-style inserts.
- Open a database connection via `connect_to_db` and delegate bulk loading
  to `load_into_table` for both exposure and metadata tables.

Conventions
-----------
- Input DataFrames are expected to have canonical column names:
  `topic_exposure_df` must include ['run_id', 'article_id', 'corpus_version',
  'topic_id', 'topic_exposure'], and `topic_metadata_df` must include
  ['run_id', 'topic_id', 'top_terms', 'cv_coherence'].
- INSERT statements use `ON CONFLICT DO NOTHING` so repeated loads with the
  same key set are safe and effectively idempotent.
- All inserts are executed within a single connection context opened by
  `connect_to_db`.

Downstream usage
----------------
Call `load_lda_tables` from modeling notebooks or batch jobs once LDA
doc-topic and topic-level artifacts have been materialized to DataFrames.
This will persist the exposures and metadata into `lda_article_topic_exposure`
and `lda_topic_metadata` for use by reporting queries and regressions.
"""

from typing import Iterable

import pandas as pd

from infra.utils.db_utils import connect_to_db, load_into_table


def load_lda_tables(
    topic_exposure_df: pd.DataFrame, topic_metadata_df: pd.DataFrame, real_run: bool = False
) -> None:
    """
    Load LDA article-topic exposures and topic metadata DataFrames into the database.

    Parameters
    ----------
    topic_exposure_df : pandas.DataFrame
        DataFrame containing per-(run_id, article_id, corpus_version, topic_id)
        topic exposure values to be inserted into `lda_article_topic_exposure`.
    topic_metadata_df : pandas.DataFrame
        DataFrame containing per-(run_id, topic_id) topic metadata, including
        `top_terms` and `cv_coherence`, to be inserted into `lda_topic_metadata`.

    Returns
    -------
    None
        The function performs inserts as a side effect and does not return a value.

    Raises
    ------
    Exception
        Any exception raised by `connect_to_db` or `load_into_table` will
        propagate to the caller, for example connection failures or SQL
        execution errors.

    Notes
    -----
    - This helper generates the appropriate INSERT statements and row iterables,
      then opens a single connection and loads both tables sequentially.
    - INSERTs are performed with `ON CONFLICT DO NOTHING`, so rerunning the
      function with overlapping keys is safe and will not duplicate rows.
    """

    if not real_run:
        return None
    lda_topic_exposure_query: str = generate_lda_article_topic_exposure_query()
    lda_topic_metadata_query: str = generate_lda_topic_metadata_query()
    lda_article_topic_exposure_rows: Iterable[tuple] = (
        create_lda_article_topic_exposure_row_generator(topic_exposure_df)
    )
    lda_topic_metadata_rows: Iterable[tuple] = create_lda_topic_metadata_row_generator(
        topic_metadata_df
    )
    with connect_to_db() as conn:
        load_into_table(conn, lda_article_topic_exposure_rows, lda_topic_exposure_query)
        load_into_table(conn, lda_topic_metadata_rows, lda_topic_metadata_query)


def generate_lda_article_topic_exposure_query() -> str:
    """
    Build the parametrized INSERT statement for `lda_article_topic_exposure`.

    Parameters
    ----------
    None

    Returns
    -------
    str
        A parametrized SQL INSERT statement that accepts a VALUES %s placeholder
        suitable for use with `load_into_table` and includes `ON CONFLICT DO NOTHING`.

    Raises
    ------
    None
        This function is pure string construction and is not expected to raise
        under normal conditions.

    Notes
    -----
    - The statement inserts into columns `(run_id, article_id, corpus_version,
      topic_id, topic_exposure)` and relies on the table's primary key and
      `ON CONFLICT DO NOTHING` to ensure idempotent loads.
    """

    return """
        INSERT INTO lda_article_topic_exposure (
        run_id,
        article_id,
        corpus_version,
        topic_id,
        topic_exposure
    ) VALUES %s ON CONFLICT DO NOTHING;
    """


def generate_lda_topic_metadata_query() -> str:
    """
    Build the parametrized INSERT statement for `lda_topic_metadata`.

    Parameters
    ----------
    None

    Returns
    -------
    str
        A parametrized SQL INSERT statement that accepts a VALUES %s placeholder
        suitable for use with `load_into_table` and includes `ON CONFLICT DO NOTHING`.

    Raises
    ------
    None
        This function is pure string construction and is not expected to raise
        under normal conditions.

    Notes
    -----
    - The statement inserts into columns `(run_id, topic_id, top_terms,
      cv_coherence)` and uses `ON CONFLICT DO NOTHING` so that repeated loads
      of the same topic keys are safe.
    """

    return """
        INSERT INTO lda_topic_metadata (
        run_id,
        topic_id,
        top_terms,
        cv_coherence
    ) VALUES %s ON CONFLICT DO NOTHING;
    """


def create_lda_article_topic_exposure_row_generator(
    topic_exposure_df: pd.DataFrame,
) -> Iterable[tuple]:
    """
    Yield row tuples for bulk-inserting article-topic exposures.

    Parameters
    ----------
    topic_exposure_df : pandas.DataFrame
        DataFrame containing the columns `run_id`, `article_id`, `corpus_version`,
        `topic_id`, and `topic_exposure` for all exposure rows to be loaded.

    Returns
    -------
    Iterable[tuple]
        An iterable of tuples, one per row in `topic_exposure_df`, ordered as
        (run_id, article_id, corpus_version, topic_id, topic_exposure) for use
        with the `lda_article_topic_exposure` INSERT statement.

    Raises
    ------
    KeyError
        If any of the required columns are missing from `topic_exposure_df`.

    Notes
    -----
    - The generator iterates over `topic_exposure_df.iterrows()` to avoid
      materializing an intermediate list of tuples, keeping memory usage
      bounded for large exposure tables.
    """

    for _, row in topic_exposure_df.iterrows():
        yield (
            row["run_id"],
            row["article_id"],
            row["corpus_version"],
            row["topic_id"],
            row["topic_exposure"],
        )


def create_lda_topic_metadata_row_generator(topic_metadata_df: pd.DataFrame) -> Iterable[tuple]:
    """
    Yield row tuples for bulk-inserting topic-level metadata.

    Parameters
    ----------
    topic_metadata_df : pandas.DataFrame
        DataFrame containing the columns `run_id`, `topic_id`, `top_terms`, and
        `cv_coherence` for all topics to be loaded into `lda_topic_metadata`.

    Returns
    -------
    Iterable[tuple]
        An iterable of tuples, one per row in `topic_metadata_df`, ordered as
        (run_id, topic_id, top_terms, cv_coherence) for use with the
        `lda_topic_metadata` INSERT statement.

    Raises
    ------
    KeyError
        If any of the required columns are missing from `topic_metadata_df`.

    Notes
    -----
    - The generator yields rows directly from `iterrows()` to integrate cleanly
      with bulk insert helpers that expect a simple iterable of tuples.
    """

    for _, row in topic_metadata_df.iterrows():
        yield (
            row["run_id"],
            row["topic_id"],
            row["top_terms"],
            row["cv_coherence"],
        )
