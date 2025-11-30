"""
Purpose
-------
Provide utility functions for constructing cleaned reference corpora, loading LDA
topic outputs, and assembling metadata used for topic coherence analysis and
downstream regressions.

Key behaviors
-------------
- Sample and clean a news-article corpus into tokenized documents suitable for
  topic coherence estimation.
- Load topic–word weights and doc–topic exposures from parquet artifacts and
  enrich them with run-level metadata.
- Build per-run topic metadata tables (top terms and coherence scores) and
  generate diagnostic coherence plots for inspection.

Conventions
-----------
- Files are resolved relative to the project root, assuming the layout:
  `local_data/lda_results/topic_word_weights` for topic–word weights and
  `local_data/lda_results/doc_topics/training` for doc–topic exposures.
- LDA runs are identified by a `run_id` that encodes the number of topics,
  corpus version, random seed, and sample date window.
- Numeric tokens are bucketed into coarse categories ("bil", "mil", "num")
  using shared BIL/MIL thresholds to align with the main LDA preprocessing
  pipeline.

Downstream usage
----------------
Import this module from the notebook to:
- Construct a bounded reference corpus for coherence calculations that is
  consistent with the training corpus.
- Load and annotate LDA artifacts before bulk-loading them into relational
  tables such as `lda_article_topic_exposure` and `lda_topic_metadata`.
- Produce quick visual diagnostics of topic coherence distributions across
  multiple LDA runs.
"""

import datetime as dt
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# fmt: off
from notebooks_utils.data_notebooks_utils.corpus_exploration_and_cleaning_utils.\
    corpus_exploration_and_cleaning_config import BIL, CHUNK_SIZE, MIL
from notebooks_utils.data_notebooks_utils.corpus_exploration_and_cleaning_utils.\
    corpus_exploration_and_cleaning_utils import (
    canonicalize_alpha_numeric_tokens,
    clean_corpus,
    extract_token_types,
    normalize_and_tokenize_sample,
    sample_corpus_per_day,
    stem_and_remove_stop_words,
)
from notebooks_utils.modeling_notebooks_utils.coherence_measurement_utils.\
    coherence_measurement_config import (
    DEFAULT_SEED_NUMBERS,
)

from infra.logging.infra_logger import InfraLogger
from notebooks_utils.data_notebooks_utils.general_data_notebooks_utils import (
    connect_with_sqlalchemy,
)

# fmt: on


def extract_clean_corpus(
    logger: InfraLogger,
    start_date: dt.date,
    end_date: dt.date,
    subsample_size: int = 20000,
    real_run: bool = False,
) -> List[List[str]]:
    """
    Build a cleaned, tokenized reference corpus for coherence estimation from parsed_news_articles.

    Parameters
    ----------
    logger : InfraLogger
        Structured logger used to record progress and debug events during corpus extraction.
    start_date : datetime.date
        Inclusive start date for sampling articles from the trading calendar.
    end_date : datetime.date
        Inclusive end date for sampling articles from the trading calendar.
    subsample_size : int
        Maximum number of articles to draw uniformly at random from the candidate article_id set.
    real_run : bool
        Flag controlling whether to perform the database and cleaning pipeline (True) or
        return a dummy sentinel value for dry runs (False).

    Returns
    -------
    list[list[str]]
        List of cleaned token sequences, one per sampled article; returns `[[]]` when
        `real_run` is False.

    Raises
    ------
    sqlalchemy.exc.SQLAlchemyError
        If the underlying database connection or query execution fails during a real run.
    pandas.errors.PandasError
        If DataFrame construction or manipulation fails while cleaning the corpus.

    Notes
    -----
    - The function samples article_ids via `sample_corpus_per_day`, queries
      `parsed_news_articles`, deduplicates by `full_text`, and then processes the
      resulting rows in chunks of size `CHUNK_SIZE`.
    - All heavy text normalization is delegated to `extract_cleaned_corpus_chunk`, which
      applies the same cleaning pipeline used for the LDA training corpus.
    """
    if not real_run:
        return [[]]
    sample_id_set = sample_corpus_per_day(start_date, end_date, logger, real_run)
    article_sample_query = """
        SELECT *
        FROM parsed_news_articles
        WHERE article_id = ANY(%(samples)s)
        ORDER BY RANDOM()
        LIMIT %(limit)s;
    """
    engine = connect_with_sqlalchemy()
    params: Mapping[str, Any] = {"samples": list(sample_id_set), "limit": subsample_size}
    filtered_sample_df: pd.DataFrame = pd.read_sql(
        article_sample_query,
        engine,
        params=params,
    ).drop_duplicates(subset=["full_text"])
    nltk.download("stopwords", quiet=True)
    cleaned_corpus: List[List[str]] = []
    for start in range(0, len(filtered_sample_df), CHUNK_SIZE):
        end: int = min(start + CHUNK_SIZE, len(filtered_sample_df))
        corpus_chunk: pd.DataFrame = filtered_sample_df.iloc[start:end].copy()
        cleaned_corpus_chunk = extract_cleaned_corpus_chunk(corpus_chunk, logger)
        cleaned_corpus.extend(cleaned_corpus_chunk)
        del corpus_chunk
    return cleaned_corpus


def extract_cleaned_corpus_chunk(
    filtered_sample_df: pd.DataFrame, logger: InfraLogger
) -> List[List[str]]:
    """
    Apply the full token normalization and cleaning pipeline to a chunk of articles.

    Parameters
    ----------
    filtered_sample_df : pandas.DataFrame
        DataFrame containing a slice of sampled articles and their token-level metadata.
    logger : InfraLogger
        Structured logger used to record each stage of the cleaning pipeline.

    Returns
    -------
    list[list[str]]
        List of cleaned, stemmed token sequences grouped by article_id for the chunk.

    Raises
    ------
    KeyError
        If expected token columns are missing from `filtered_sample_df`.
    pandas.errors.PandasError
        If groupby or column operations fail during cleaning.

    Notes
    -----
    - The function runs the same ordered pipeline as the main LDA preprocessing:
      normalization, token-type extraction, numerical canonicalization, alpha-numeric
      canonicalization, stop-word removal, and final corpus cleaning.
    - The final grouping assumes that `article_id` and `stemmed_tokens` columns exist
      after the cleaning helpers have been applied.
    """
    logger.debug(event="canonicalize_and_tokenize_corpus")
    filtered_sample_df = normalize_and_tokenize_sample(filtered_sample_df)
    logger.debug(event="normalized_and_tokenized_sample")
    filtered_sample_df = extract_token_types(filtered_sample_df)
    logger.debug(event="extracted_token_types")
    filtered_sample_df = canonicalize_numerical_tokens_coherence(filtered_sample_df)
    logger.debug(event="canonicalized_numerical_tokens")
    filtered_sample_df = canonicalize_alpha_numeric_tokens(filtered_sample_df)
    logger.debug(event="canonicalized_alpha_numeric_tokens")
    filtered_sample_df = stem_and_remove_stop_words(filtered_sample_df)
    logger.debug(event="stemmed_and_removed_stop_words")
    filtered_sample_df = clean_corpus(filtered_sample_df)
    logger.debug(event="cleaned_corpus")
    return filtered_sample_df.groupby("article_id")["stemmed_tokens"].apply(list).tolist()


def canonicalize_numerical_tokens_coherence(filtered_sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw numeric tokens into coarse canonical bins used for coherence estimation.

    Parameters
    ----------
    filtered_sample_df : pandas.DataFrame
        Token-level DataFrame containing at least `token_type` and `raw_tokens`
        columns, and optionally existing numeric metadata.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with additional columns `numeric_value`, `numerical_token_key`,
        and `canonical_numerical_token` added or updated.

    Raises
    ------
    ValueError
        If numeric conversion of `raw_tokens` fails in an unexpected way.
    pandas.errors.PandasError
        If column assignment or selection fails during canonicalization.

    Notes
    -----
    - Numeric tokens are converted to absolute float values and bucketed based on the
      configured `BIL` and `MIL` thresholds.
    - The canonical categories are:
      1 → "bil" (billion-scale), 2 → "mil" (million-scale), 3 → "num" (all other numerics),
      with non-numeric tokens receiving no canonical label.
    """
    numerical_token_mask = filtered_sample_df["token_type"] == "numeric"
    filtered_sample_df.loc[numerical_token_mask, "numeric_value"] = pd.to_numeric(
        filtered_sample_df.loc[numerical_token_mask, "raw_tokens"],
        errors="coerce",
    )
    numeric_vals = filtered_sample_df["numeric_value"].abs()
    conditions = [
        numerical_token_mask & (numeric_vals >= BIL),
        numerical_token_mask & (numeric_vals >= MIL) & (numeric_vals < BIL),
        numerical_token_mask & (numeric_vals < MIL),
    ]
    numerical_keys: List[int] = [1, 2, 3]
    filtered_sample_df["numerical_token_key"] = np.select(
        conditions,
        numerical_keys,
        default=0,
    )
    numerical_token_types = {
        1: "bil",
        2: "mil",
        3: "num",
    }
    filtered_sample_df["canonical_numerical_token"] = filtered_sample_df["numerical_token_key"].map(
        numerical_token_types
    )

    return filtered_sample_df


def extract_word_weight_dfs() -> List[pd.DataFrame]:
    """
    Load all topic-word weight parquet files for K=200 LDA runs into DataFrames.

    Parameters
    ----------
    None

    Returns
    -------
    list[pandas.DataFrame]
        List of DataFrames, one per parquet file, each containing topic-word weight
        information as emitted by MALLET.

    Raises
    ------
    OSError
        If the topic-word weight directory does not exist or cannot be accessed.
    pandas.errors.PandasError
        If reading any parquet file fails.

    Notes
    -----
    - The search root is computed relative to this file, assuming the standard
      `local_data/lda_results/topic_word_weights` layout.
    - Only files matching the pattern `K200_seed*.parquet` are loaded, one per
      independent LDA run.
    """
    project_root: Path = Path(__file__).resolve().parents[4]
    topic_word_weight_dir: Path = project_root / "local_data" / "lda_results" / "topic_word_weights"
    word_weight_dfs: List[pd.DataFrame] = []
    for topic_word_weight_file in topic_word_weight_dir.glob("K200_seed*.parquet"):
        topic_word_weight_df: pd.DataFrame = pd.read_parquet(topic_word_weight_file)
        word_weight_dfs.append(topic_word_weight_df)
    return word_weight_dfs


def extract_top_words(
    word_weight_dfs: List[pd.DataFrame], top_word_amount: int = 10
) -> List[List[List[str]]]:
    """
    Extract top-N terms per topic for each LDA run from topic-word weight DataFrames.

    Parameters
    ----------
    word_weight_dfs : list[pandas.DataFrame]
        Sequence of DataFrames containing `topic_id`, `term`, and `weight` columns
        for one or more LDA runs.
    top_word_amount : int
        Number of highest-weight terms to retain per topic in each run.

    Returns
    -------
    list[list[list[str]]]
        Nested list structure where the outer index selects the run, the middle
        index selects the topic within that run, and the innermost list contains
        the ordered top terms for that topic.

    Raises
    ------
    KeyError
        If any DataFrame is missing `topic_id`, `term`, or `weight` columns.
    pandas.errors.PandasError
        If groupby, nlargest, or sorting operations fail while extracting terms.

    Notes
    -----
    - For each run, the function groups by `topic_id`, takes the `top_word_amount`
      largest weights, and sorts terms by decreasing weight within each topic.
    - The returned term order matches the order expected for topic coherence
      calculations and downstream inspection.
    """
    top_words_list: List[List[List[str]]] = []
    for word_weight_df in word_weight_dfs:
        top_n_words_mask: pd.Series = word_weight_df.groupby(["topic_id"])["weight"].nlargest(
            top_word_amount
        )
        top_n_words_df: pd.DataFrame = word_weight_df.loc[
            top_n_words_mask.index.get_level_values(1)
        ]
        top_words_list.append(
            top_n_words_df.sort_values(by=["topic_id", "weight"], ascending=[True, False])
            .groupby("topic_id")["term"]
            .apply(list)
            .tolist()
        )
    return top_words_list


def plot_coherence(coherence_scores: List[List[float]]) -> None:
    """
    Plot per-topic C_v coherence distributions and scatter clusters for multiple runs.

    Parameters
    ----------
    coherence_scores : list[list[float]]
        Nested list where each outer element corresponds to one LDA run and each
        inner list contains per-topic C_v coherence scores for that run.

    Returns
    -------
    None
        The function produces matplotlib plots as a side effect and prints a
        summary table to stdout.

    Raises
    ------
    ValueError
        If `coherence_scores` is empty or has inconsistent inner lengths that
        prevent DataFrame construction.
    RuntimeError
        If matplotlib fails to create or render the plots.

    Notes
    -----
    - The function constructs a long-form DataFrame of coherence values, prints
      summary statistics per run, and then produces:
        (1) a boxplot of coherence distributions per run and
        (2) a scatter plot of individual topic scores by run.
    """
    records = []
    for run_idx, scores in enumerate(coherence_scores):
        for topic_idx, score in enumerate(scores):
            records.append({"run_id": run_idx, "topic_id": topic_idx, "cv_coherence": score})

    coh_df = pd.DataFrame.from_records(records)

    # 2) Summary table per run
    summary: pd.DataFrame = (
        coh_df.groupby("run_id")["cv_coherence"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    print(summary)

    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot(
        [coh_df[coh_df["run_id"] == run_id]["cv_coherence"] for run_id in summary["run_id"]],
        labels=[f"Run {run_id}" for run_id in summary["run_id"]],
    )
    axes[0].set_title("CV Coherence Distribution per Run")
    axes[0].set_ylabel("CV Coherence")

    # coherence clusters plot
    for run_id in summary["run_id"]:
        run_coh_scores: pd.Series = coh_df[coh_df["run_id"] == run_id]["cv_coherence"]
        axes[1].scatter(
            [run_id] * len(run_coh_scores),
            run_coh_scores,
            alpha=0.6,
        )
    axes[1].set_title("CV Coherence Scores per Topic")
    axes[1].set_xlabel("Run ID")
    axes[1].set_ylabel("CV Coherence")

    plt.tight_layout()
    plt.show()


def extract_topic_exposure_dfs(
    sample_start: dt.date, sample_end: dt.date, corpus_version: int = 1
) -> List[pd.DataFrame]:
    """
    Load MALLET doc-topics parquet files and annotate them with run and corpus metadata.

    Parameters
    ----------
    sample_start : datetime.date
        Inclusive start date used to label the temporal window for the LDA run.
    sample_end : datetime.date
        Inclusive end date used to label the temporal window for the LDA run.
    corpus_version : int
        Integer identifying the corpus version associated with these exposures.

    Returns
    -------
    list[pandas.DataFrame]
        List of DataFrames, one per doc-topics parquet file, each augmented with
        `run_id` and `corpus_version` columns.

    Raises
    ------
    OSError
        If the topic exposure directory cannot be accessed or globbing fails.
    pandas.errors.PandasError
        If reading any parquet file or assigning new columns fails.

    Notes
    -----
    - The function expects doc-topics files under
      `local_data/lda_results/doc_topics/training` with filenames matching
      `K200_seed*.parquet`.
    - The `run_id` string encodes the corpus version, seed number, and sample
      date window to support downstream auditability.
    """
    project_root: Path = Path(__file__).resolve().parents[4]
    topic_exposure_dir: Path = (
        project_root / "local_data" / "lda_results" / "doc_topics" / "training"
    )
    topic_exposure_dfs: List[pd.DataFrame] = []
    for topic_exposure_file in topic_exposure_dir.glob("K200_seed*.parquet"):
        seed_number = str(topic_exposure_file).rsplit("/", maxsplit=1)[-1][9:11]
        topic_exposure_df: pd.DataFrame = pd.read_parquet(topic_exposure_file)
        topic_exposure_df["run_id"] = (
            f"K200_v{corpus_version}_seed{seed_number}_{sample_start}_{sample_end}"
        )
        topic_exposure_df["corpus_version"] = corpus_version
        topic_exposure_df.rename(
            columns={"instance_id": "article_id", "topic_proportion": "topic_exposure"},
            inplace=True,
        )
        topic_exposure_dfs.append(topic_exposure_df)
    return topic_exposure_dfs


def generate_topic_metadata_dfs(
    sample_start_date: dt.date,
    sample_end_date: dt.date,
    top_words_list: List[List[List[str]]],
    coherence_scores: List[List[float]],
    seed_numbers: Iterable[int] | None = None,
    corpus_version: int = 1,
) -> List[pd.DataFrame]:
    """
    Construct per-run topic metadata DataFrames from top terms and coherence scores.

    Parameters
    ----------
    sample_start_date : datetime.date
        Inclusive start date of the sample window used to train each LDA run.
    sample_end_date : datetime.date
        Inclusive end date of the sample window used to train each LDA run.
    top_words_list : list[list[list[str]]]
        Nested list of top terms; outer index selects the run, middle index selects
        the topic, and inner lists contain ordered top terms for that topic.
    coherence_scores : list[list[float]]
        Nested list of C_v coherence scores aligned with `top_words_list` for each
        run and topic.
    seed_numbers : list[int] or None
        Optional list of seed integers used for each run; if None, `DEFAULT_SEED_NUMBERS`
        is used.
    corpus_version : int
        Integer identifying the corpus version associated with these topics.

    Returns
    -------
    list[pandas.DataFrame]
        List of topic metadata DataFrames, one per run, each containing columns
        `run_id`, `topic_id`, `top_terms`, and `cv_coherence`.

    Raises
    ------
    ValueError
        If the lengths of `top_words_list`, `coherence_scores`, and `seed_numbers`
        are inconsistent.
    pandas.errors.PandasError
        If DataFrame construction fails for any run.

    Notes
    -----
    - The `run_id` for each run encodes the corpus version, seed number, and
      sample date window, matching the identifiers used in topic exposure tables.
    """
    topic_metadata_dfs: List[pd.DataFrame] = []
    if seed_numbers is None:
        seed_numbers = DEFAULT_SEED_NUMBERS
    for _, (top_words, coh_scores, seed_number) in enumerate(
        zip(top_words_list, coherence_scores, seed_numbers)
    ):
        run_id = f"K200_v{corpus_version}_seed{seed_number}_{sample_start_date}_{sample_end_date}"
        topic_metadata_df = pd.DataFrame(
            {
                "run_id": [run_id] * len(top_words),
                "topic_id": list(range(len(top_words))),
                "top_terms": top_words,
                "cv_coherence": coh_scores,
            }
        )
        topic_metadata_dfs.append(topic_metadata_df)
    return topic_metadata_dfs
