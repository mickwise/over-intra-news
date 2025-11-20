"""
Purpose
-------
Load CC-NEWS-derived Parquet outputs (articles and per-session sample stats)
from the local filesystem into the Postgres research database.

Key behaviors
-------------
- Walks the `local_data/ccnews_articles` hierarchy and loads all
  `articles.parquet` files into the `parsed_news_articles` table via
  `load_parsed_news_articles`.
- Walks the `local_data/ccnews_sample_stats` hierarchy, aggregates raw
  per-sample stats into one row per (trading_day, session), and inserts
  them into the `news_sample_stats` table.
- Reads database credentials from environment variables (optionally via
  `.env` using `python-dotenv`) and uses a single connection per stats
  loading run.

Conventions
-----------
- Directory layout is assumed to follow:
    - `local_data/ccnews_articles/year=YYYY/month=MM/day=DD/session=SESSION/articles.parquet`
    - `local_data/ccnews_sample_stats/year=YYYY/month=MM
      /day=DD/session=SESSION/sample_stats.parquet`
- `articles.parquet` is expected to contain:
    - `ny_date` (trading date in America/New_York),
    - `word_count`, `session`, `cik_list`, and other article fields
      required by `load_parsed_news_articles`.
- `sample_stats.parquet` is expected to contain:
    - `date` (canonical trading date),
    - count columns like `records_scanned`, `html_200_count`,
      `ge_25_words`, `articles_kept`, etc.
- Inserts into `news_sample_stats` use `ON CONFLICT (trading_day, session) DO NOTHING`
  so that re-running the loader is idempotent for already-populated
  buckets.

Downstream usage
----------------
- Run as a script from the project root once Parquet outputs are
  available and the Postgres instance is reachable:

    `python -m aws.loading.load_articles_and_sample_stats`

- Other code should generally treat this as a one-off or periodic ETL
  utility rather than importing its helpers directly.
"""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from aws.loading.load_parsed_news_articles import load_parsed_news_articles
from infra.utils.db_utils import connect_to_db

BASE_ARTICLES = Path("local_data/ccnews_articles")
BASE_STATS = Path("local_data/ccnews_sample_stats")


def main() -> None:
    """
    Orchestrate loading of articles and sample stats from Parquet into Postgres.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    psycopg2.OperationalError
        If the database connection cannot be established using the configured
        environment variables.
    FileNotFoundError
        Indirectly, if expected Parquet files are missing and their absence
        is not guarded by existence checks (articles are guarded; stats rely
        on `parquet_path.exists()` as well).

    Notes
    -----
    - Calls `load_dotenv()` to hydrate environment variables from a `.env`
      file before opening any database connections.
    - Invokes `load_all_articles()` first, then `load_all_sample_stats()`,
      so articles are populated before aggregate stats.
    """
    load_dotenv()
    load_all_articles()
    load_all_sample_stats()


def load_all_articles() -> None:
    """
    Walk the local articles directory tree and load all article Parquet files.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    OSError
        If there are issues reading Parquet files from disk.
    psycopg2.Error
        If an error occurs while inserting into `parsed_news_articles`.

    Notes
    -----
    - Expects `BASE_ARTICLES` to point to a directory organized as:
        `year=YYYY/month=MM/day=DD/session=SESSION/articles.parquet`.
    - For each `(year, month, day, session)` combination:
        - Skips the session if `articles.parquet` is missing.
        - Converts `ny_date` to a Python `date` object to match the
          `DATE` column type in Postgres.
        - Delegates bulk insertion to `load_parsed_news_articles(...)`,
          which uses batched `execute_values` calls.
    """
    for year_dir in sorted(BASE_ARTICLES.glob("year=*")):
        for month_dir in sorted(year_dir.glob("month=*")):
            for day_dir in sorted(month_dir.glob("day=*")):
                for session in ("intraday", "overnight"):
                    session_dir = day_dir / f"session={session}"
                    parquet_path = session_dir / "articles.parquet"
                    if not parquet_path.exists():
                        continue
                    articles_df: pd.DataFrame = pd.read_parquet(parquet_path)
                    articles_df = articles_df[articles_df["word_count"] >= 25]
                    articles_df["ny_date"] = pd.to_datetime(articles_df["ny_date"]).dt.date
                    load_parsed_news_articles(articles_df)


def load_all_sample_stats() -> None:
    """
    Aggregate per-sample stats into per-(trading_day, session) rows and load them.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    psycopg2.Error
        If insertion into `news_sample_stats` fails for any row.
    OSError
        If there are issues reading `sample_stats.parquet` files from disk.

    Notes
    -----
    - Uses a single database connection and cursor for the entire traversal
      of `BASE_STATS`.
    - For each `(year, month, day, session)` combination:
        - Skips the session if `sample_stats.parquet` is missing.
        - Assumes all rows share the same `date` value and uses the first
          entry as the canonical `trading_day`.
        - Sums each count-type column across all samples for that
          `(date, session)` and inserts one aggregated row into
          `news_sample_stats`.
    - Inserts are idempotent at the (trading_day, session) level due to
      `ON CONFLICT (trading_day, session) DO NOTHING`.
    """
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            for year_dir in sorted(BASE_STATS.glob("year=*")):
                for month_dir in sorted(year_dir.glob("month=*")):
                    for day_dir in sorted(month_dir.glob("day=*")):
                        for session in ("intraday", "overnight"):
                            session_dir = day_dir / f"session={session}"
                            parquet_path = session_dir / "sample_stats.parquet"
                            if not parquet_path.exists():
                                continue
                            sample_stats_df: pd.DataFrame = pd.read_parquet(parquet_path)
                            trading_day = pd.to_datetime(sample_stats_df["date"].iloc[0]).date()
                            aggragated_data: tuple = (
                                trading_day,
                                session,
                                int(sample_stats_df["records_scanned"].sum()),
                                int(sample_stats_df["html_200_count"].sum()),
                                int(sample_stats_df["unhandled_errors"].sum()),
                                int(sample_stats_df["decompression_errors"].sum()),
                                int(sample_stats_df["ge_25_words"].sum()),
                                int(sample_stats_df["too_long_articles"].sum()),
                                int(sample_stats_df["english_count"].sum()),
                                int(sample_stats_df["matched_any_firm"].sum()),
                                int(sample_stats_df["articles_kept"].sum()),
                            )
                            query: str = generate_news_sample_stats_query()
                            cursor.execute(query, aggragated_data)
        conn.commit()


def generate_news_sample_stats_query() -> str:
    """
    Build the parameterized INSERT statement for `news_sample_stats`.

    Parameters
    ----------
    None

    Returns
    -------
    str
        A parameterized SQL string suitable for `cursor.execute(...)` with
        a single value tuple `(trading_day, session, ...)` covering all
        stats fields.

    Raises
    ------
    None

    Notes
    -----
    - Uses `ON CONFLICT (trading_day, session) DO NOTHING` so that re-running
      the loader does not overwrite or error on existing rows for the same
      (trading_day, session).
    - The placeholder order must match the tuple constructed in
      `load_all_sample_stats()`.
    """
    return """
    INSERT INTO news_sample_stats (
        trading_day,
        session,
        records_scanned,
        html_200_count,
        unhandled_errors,
        decompression_errors,
        ge_25_words,
        too_long_articles,
        english_count,
        matched_any_firm,
        articles_kept
    ) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (trading_day, session) DO NOTHING;
    """


if __name__ == "__main__":
    main()
