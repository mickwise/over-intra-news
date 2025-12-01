"""
Purpose
-------
Load MALLET LDA *inference* doc–topic Parquet files from
`local_data/lda_results/doc_topics/inference` and persist them into the
relational store as `lda_article_topic_exposure` rows.

Key behaviors
-------------
- For each configured seed in `DEFAULT_SEED_NUMBERS`, read the corresponding
  inference Parquet file (one row per `(doc_index, instance_id, topic_id)`).
- Convert those rows into `(run_id, article_id, corpus_version, topic_id,
  topic_exposure)` tuples compatible with `lda_article_topic_exposure`.
- Open a single database connection per seed and bulk-insert using
  `load_into_table` with `ON CONFLICT DO NOTHING`, so reruns are idempotent.

Conventions
-----------
- Expects Parquet files named `K200_seed{seed}.parquet` under
  `local_data/lda_results/doc_topics/inference`, conforming to the
  `lda_inference_doc_topics` schema (`doc_index`, `instance_id`,
  `topic_id`, `topic_proportion`).
- `instance_id` is treated as the `article_id` key used in `lda_documents`.
- `run_id` encodes the corpus version, random seed, and the *training*
  date window, e.g. `K200_v1_seed42_2022-08-01_2025-08-01`.
- Only inference exposures are loaded here; training doc–topic exposures
  are loaded elsewhere.

Downstream usage
----------------
Call `load_inference_topic_exposures` once the inference doc–topic Parquets
have been synced to `local_data/`, to materialize out-of-sample article–topic
exposures for regressions and portfolio construction.
"""

import datetime as dt
from pathlib import Path
from typing import Iterator

import pandas as pd
from dotenv import load_dotenv

# fmt: off
from notebooks_utils.modeling_notebooks_utils.coherence_measurement_utils.\
    coherence_measurement_config import (
    DEFAULT_SEED_NUMBERS,
)

from infra.utils.db_utils import connect_to_db, load_into_table
from notebooks_utils.modeling_notebooks_utils.coherence_measurement_utils.lda_loading import (
    generate_lda_article_topic_exposure_query,
)

# fmt: on


def load_inference_topic_exposures(
    training_start_date: dt.date = dt.date(2016, 8, 1),
    training_end_date: dt.date = dt.date(2022, 8, 1),
    corpus_version: int = 1,
) -> None:
    """
    Load per-article LDA topic exposures for the inference corpus into
    `lda_article_topic_exposure` for all configured seeds.

    Parameters
    ----------
    training_start_date : datetime.date, optional
        Start date of the *training* window used when fitting the LDA model.
        Used here only to reconstruct the `run_id` string so that inference
        exposures share the same run identifier as the training exposures.
    training_end_date : datetime.date, optional
        End date (exclusive) of the training window. Must match the value used
        when the corresponding LDA model was trained.
    corpus_version : int, optional
        Logical corpus version identifier shared with `lda_documents` and used
        in the `lda_article_topic_exposure` primary key. Defaults to 1.

    Returns
    -------
    None
        Performs bulk inserts into Postgres as a side effect; no value is
        returned.

    Raises
    ------
    FileNotFoundError
        If a required Parquet file for a configured seed is missing.
    Exception
        Any connection or SQL execution error propagated from `connect_to_db`
        or `load_into_table`.

    Notes
    -----
    - Iterates over `DEFAULT_SEED_NUMBERS` and, for each seed, reads the
      corresponding inference doc–topics Parquet file
      (`K200_seed{seed}.parquet`).
    - Constructs a `run_id` of the form
      `K200_v{corpus_version}_seed{seed}_{training_start_date}_{training_end_date}`,
      matching the convention used on the training side.
    - Only one inference DataFrame is held in memory at a time; rows are
      streamed to the database via a generator, so the total memory footprint
      is bounded even though each Parquet file contains tens of millions of
      `(article, topic)` pairs.
    """
    load_dotenv()
    lda_article_topic_exposure_query: str = generate_lda_article_topic_exposure_query()
    for seed_number in DEFAULT_SEED_NUMBERS:
        current_run_id: str = (
            f"K200_v{corpus_version}_seed{seed_number}_"
            f'{training_start_date.strftime("%Y-%m-%d")}_'
            f'{training_end_date.strftime("%Y-%m-%d")}'
        )
        project_root: Path = Path(__file__).resolve().parents[2]
        current_inference_topic_exposures_path: Path = (
            project_root / "local_data" / "lda_results" / "doc_topics" / "inference"
        )
        current_inference_topic_exposures_file: Path = (
            current_inference_topic_exposures_path / f"K200_seed{seed_number}.parquet"
        )
        current_inference_topic_exposure_df: pd.DataFrame = pd.read_parquet(
            current_inference_topic_exposures_file
        )
        current_row_generator: Iterator[tuple] = (
            create_current_inference_topic_exposure_row_generator(
                current_inference_topic_exposure_df,
                current_run_id,
                corpus_version,
            )
        )
        with connect_to_db() as conn:
            load_into_table(
                conn,
                current_row_generator,
                lda_article_topic_exposure_query,
            )


def create_current_inference_topic_exposure_row_generator(
    topic_exposure_df: pd.DataFrame,
    current_run_id: str,
    corpus_version: int = 1,
) -> Iterator[tuple]:
    """
    Yield row tuples for bulk-inserting inference article–topic exposures.

    Parameters
    ----------
    topic_exposure_df : pandas.DataFrame
        DataFrame containing the columns `instance_id`, `topic_id`, and
        `topic_proportion` for a single inference doc–topics batch, typically
        read from `K200_seed{seed}.parquet`.
    current_run_id : str
        Fully-formed LDA run identifier (e.g.
        `K200_v1_seed42_2016-08-01_2022-08-01`) shared with the corresponding
        training exposures.
    corpus_version : int, optional
        Corpus version identifier to stamp into `lda_article_topic_exposure`.
        Defaults to 1.

    Returns
    -------
    Iterator[tuple]
        An iterator of tuples
        `(run_id, article_id, corpus_version, topic_id, topic_exposure)`
        matching the parameter order in `generate_lda_article_topic_exposure_query()`.

    Raises
    ------
    KeyError
        If any of the expected columns are missing from `topic_exposure_df`.

    Notes
    -----
    - `instance_id` is treated as `article_id` and is assumed to match
      `lda_documents.article_id` for the given `corpus_version`.
    - The generator streams rows directly from `topic_exposure_df.iterrows()`
      so callers do not need to build an intermediate list of tuples. Once the
      bulk load for a given seed completes and the DataFrame goes out of
      scope, it becomes eligible for garbage collection.
    """
    for _, row in topic_exposure_df.iterrows():
        yield (
            current_run_id,
            row["instance_id"],
            corpus_version,
            row["topic_id"],
            row["topic_proportion"],
        )
