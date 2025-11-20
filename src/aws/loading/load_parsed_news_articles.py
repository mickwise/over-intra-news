"""
Purpose
-------
Load cleaned, firm-linked CC-NEWS article records from a pandas DataFrame
into the `parsed_news_articles` fact table in PostgreSQL.

Key behaviors
-------------
- Build a parameterized INSERT ... VALUES %s statement targeting the
  `parsed_news_articles` schema.
- Stream DataFrame rows as Python tuples into `load_into_table(...)`,
  which batches inserts using `psycopg2.extras.execute_values`.
- Coerce `cik_list` values into plain Python lists so they map cleanly
  onto PostgreSQL text[] columns.

Conventions
-----------
- The input DataFrame is expected to mirror the Parquet schema produced
  by the CC-NEWS parsing pipeline, with at least:
  `article_id`, `ny_date`, `session`, `cik_list`, `warc_path`,
  `warc_date_utc`, `url`, `http_status`, `http_content_type`,
  `word_count`, `language_confidence`, and `full_text`.
- `ny_date` is the NYSE trading date and is passed through as
  `trading_day` to the database; it is already be normalized to a
  `datetime.date` where possible.
- Uniqueness is enforced on `article_id`; conflicts are ignored via
  `ON CONFLICT (article_id) DO NOTHING`.

Downstream usage
----------------
Typical usage is from a higher-level loader that:
- Reads `articles.parquet` files into a DataFrame,
- Applies any last-mile filters (e.g., `word_count >= 25`,
  trading-date remapping),
- Invokes `load_parsed_news_articles(df)` to persist the records.
"""

import pandas as pd

from infra.utils.db_utils import connect_to_db, load_into_table


def load_parsed_news_articles(article_df: pd.DataFrame) -> None:
    """
    Stream article records from a DataFrame into `parsed_news_articles`.

    Parameters
    ----------
    article_df : pandas.DataFrame
        DataFrame containing one row per article with columns compatible
        with the `parsed_news_articles` INSERT statement produced by
        `generate_parsed_news_articles_query()`.

    Returns
    -------
    None
        All rows are sent to PostgreSQL in batches; no value is returned.

    Raises
    ------
    psycopg2.DatabaseError
        Propagated if the underlying connection, cursor, or batched insert
        fails for any reason (network issues, constraint violations that
        are not handled by `ON CONFLICT`, etc.).
    KeyError
        If required columns are missing from `article_df`.

    Notes
    -----
    - Uses `connect_to_db()` to open a short-lived PostgreSQL connection.
    - Delegates batching and `execute_values` calls to `load_into_table`.
    - The caller is responsible for ensuring that input data respects the
      table-level CHECK constraints (e.g., `word_count >= 25`,
      `language_confidence >= 0.99`).
    """
    query: str = generate_parsed_news_articles_query()
    generator = create_article_row_generator(article_df)
    with connect_to_db() as conn:
        load_into_table(conn, generator, query)


def generate_parsed_news_articles_query() -> str:
    """
    Build the INSERT statement for `parsed_news_articles`.

    Parameters
    ----------
    None

    Returns
    -------
    str
        A parameterized INSERT statement compatible with
        `psycopg2.extras.execute_values`, containing a single `%s`
        placeholder for the batched VALUES.

    Raises
    ------
    None

    Notes
    -----
    - The column order here must exactly match the tuples produced by
      `create_article_row_generator(...)`.
    - `ON CONFLICT (article_id) DO NOTHING` ensures the loader is
      idempotent with respect to previously ingested articles.
    """
    return """
    INSERT INTO parsed_news_articles (
        article_id,
        trading_day,
        session,
        cik_list,
        warc_path,
        warc_date_utc,
        url,
        http_status,
        http_content_type,
        word_count,
        language_confidence,
        full_text
    ) VALUES %s ON CONFLICT (article_id) DO NOTHING;
    """


def create_article_row_generator(article_df: pd.DataFrame):
    """
    Yield per-row value tuples for insertion into `parsed_news_articles`.

    Parameters
    ----------
    article_df : pandas.DataFrame
        DataFrame with one row per article and at least the following
        columns: `article_id`, `ny_date`, `session`, `cik_list`,
        `warc_path`, `warc_date_utc`, `url`, `http_status`,
        `http_content_type`, `word_count`, `language_confidence`,
        `full_text`.

    Returns
    -------
    Iterable[tuple]
        A generator yielding tuples in the exact order expected by the
        `parsed_news_articles` INSERT statement.

    Raises
    ------
    KeyError
        If any required column is missing from `article_df`.

    Notes
    -----
    - `cik_list` values are explicitly converted to `list(...)` so that
      numpy arrays or pandas extension dtypes are coerced into plain
      Python lists, which psycopg2 can adapt to PostgreSQL text[].
    - Assumes `ny_date` is already a `datetime.date` (or a string that
      psycopg2 can cast to DATE); upstream loaders are responsible for
      any timezone normalization and mapping from WARC capture date to
      NYSE trading date.
    """
    for _, row in article_df.iterrows():
        yield (
            row["article_id"],
            row["ny_date"],
            row["session"],
            list(row["cik_list"]),
            row["warc_path"],
            row["warc_date_utc"],
            row["url"],
            row["http_status"],
            row["http_content_type"],
            row["word_count"],
            row["language_confidence"],
            row["full_text"],
        )
