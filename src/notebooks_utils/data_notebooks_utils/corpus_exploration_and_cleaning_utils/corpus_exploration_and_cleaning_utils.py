"""
Purpose
-------
Provide the full corpus-sampling, cleaning, tokenization, and frequency-estimation
pipeline used to construct versioned LDA training corpora. This includes sampling
news articles from Postgres, applying multi-stage normalization and token
canonicalization, writing tokenized parquet chunks, and aggregating global and
document-level term statistics.

Key behaviors
-------------
- Pulls trading-day–aligned news articles from Postgres.
- Applies deterministic cleaning, canonicalization, and stemming rules.
- Writes tokenized corpus chunks to parquet for scalable downstream use.
- Computes all token/document counters needed for LDA vocabulary and
  document–term matrix construction.

Conventions
-----------
-- All article text is uppercased upstream.
- All text normalization is deterministic and stateless (no randomness).
- Tokenization operates on uppercase-normalized ASCII text.
- Numeric tokens are mapped into canonical magnitude buckets (__NUM__/__MIL__/__BIL__).
- Intermediate parquet chunks are named `tokenized_corpus_chunk_*.parquet`.
- All date handling is in UTC; trading days come from `trading_calendar`.

Downstream usage
----------------
Used by the LDA ingestion module to populate:
- `lda_documents`
- `lda_vocabulary`
- `lda_document_terms`

Downstream modules call this to (1) generate tokenized parquet, and
(2) compute the `FrequencyCounters` object used for DB ingestion.
"""

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Counter, List

import nltk
import numpy as np
import pandas as pd
import sqlalchemy as sa

# fmt: on
from infra.logging.infra_logger import InfraLogger

# fmt: off
from notebooks_utils.data_notebooks_utils.\
    corpus_exploration_and_cleaning_utils.corpus_exploration_and_cleaning_config import (
    BIL,
    CHUNK_SIZE,
    MIL,
    MINIMAL_CHAR_COUNT_PER_TOKEN,
    NOISY_PREFIXES,
    NOISY_SUBSTRINGS,
    STRONG_ENGLISH_CONFIDENCE_THRESHOLD,
    TOKENIZED_PARQUET_DIR,
)
from notebooks_utils.data_notebooks_utils.general_data_notebooks_utils import (
    connect_with_sqlalchemy,
)


@dataclass
class FrequencyCounters:
    """
    Purpose
    -------
    Represent the aggregated frequency statistics for the active LDA corpus view.
    Holds token-level and document-level counters that are passed to
    the DB ingestion layer.

    Key behaviors
    -------------
    - Stores per-token global frequency and document frequency.
    - Optionally stores per-document total/unique token counts.
    - Optionally stores within-document term counts for lda_document_terms.
    - After pruning, all non-None counters are defined with respect to the same
      active vocabulary.

    Parameters
    ----------
    token_frequency_counter : Counter[str]
        Global term counts for each token in the active vocabulary.
    token_document_counter : Counter[str]
        Document-frequency counts for each token in the active vocabulary.
    document_global_counter : Counter[str] | None
        Optional mapping article_id → total token count after pruning. When None,
        document-level counts have not been computed yet.
    document_unique_counter : Counter[str] | None
        Optional mapping article_id → number of unique tokens after pruning. When
        None, document-level counts have not been computed yet.
    token_within_document_counter : Counter[tuple[str, str]] | None
        Optional mapping (article_id, token) → term_count. When None, per-document
        term counts have not been computed yet.

    Attributes
    ----------
    token_frequency_counter : Counter[str]
        Global token frequencies for the current corpus_version and pruning rules.
    token_document_counter : Counter[str]
        Document-frequency counts for the current corpus_version and pruning rules.
    document_global_counter : Counter[str] | None
        Per-document global token counts, or None if not yet populated.
    document_unique_counter : Counter[str] | None
        Per-document unique token counts, or None if not yet populated.
    token_within_document_counter : Counter[(str, str)] | None
        Per-(document, token) term counts, or None if not yet populated.

    Notes
    -----
    - The typical flow is:
    1. Build token-level counters from parquet (extract_token_distributions).
    2. Prune the vocabulary and compute document-level counters
        (summarize_and_filter_vocabulary + extract_per_document_per_term_counters).
    - After pruning, all non-None counters should be internally consistent and
      refer to the same token set.
    """

    token_frequency_counter: Counter[str]
    token_document_counter: Counter[str]
    document_global_counter: Counter[str] | None = None
    document_unique_counter: Counter[str] | None = None
    token_within_document_counter: Counter[tuple[str, str]] | None = None


def sample_corpus_per_day(
    start_date: dt.date, end_date: dt.date, logger: InfraLogger, real_run: bool = False
) -> set[str]:
    """
    Sample all article_ids between two dates that satisfy boilerplate removal and
    language-quality filters.

    Parameters
    ----------
    start_date : datetime.date
        First trading day (inclusive) to sample.
    end_date : datetime.date
        Last trading day (inclusive) to sample.
    logger : InfraLogger
        Structured logger used for debug-level sampling events.
    real_run : bool
        If False, the function returns an empty set without hitting the database.

    Returns
    -------
    set[str]
        Set of article_ids passing basic quality/boilerplate filters across all
        trading days and sessions.

    Notes
    -----
    - Only trading days with is_trading_day=TRUE are used.
    - Both “overnight” and “intraday” sessions are sampled for each day.
    - Sampling occurs before tokenization and produces a unique set of article_ids.
    """

    if not real_run:
        return set()
    trading_calendar_query: str = """
    SELECT trading_day
    FROM trading_calendar
    WHERE trading_day BETWEEN %s AND %s
    AND is_trading_day = TRUE;
    """

    engine: sa.Engine = connect_with_sqlalchemy()
    article_samples: set[str] = set()
    with engine.connect() as conn:
        trading_days_df: pd.DataFrame = pd.read_sql(
            trading_calendar_query, conn, params=(start_date, end_date)
        )
        for day in trading_days_df["trading_day"]:
            for session in ["overnight", "intraday"]:
                logger.debug(
                    event="sample_corpus_per_day", context={"day": day, "session": session}
                )
                corpus_query: str = """
                SELECT 
                article_id,
                language_confidence,
                full_text
                FROM parsed_news_articles
                WHERE trading_day = %s AND session = %s;
                """
                daily_sample_df: pd.DataFrame = pd.read_sql(
                    corpus_query, conn, params=(day, session)
                )
                article_samples.update(sample_per_day_session(daily_sample_df))
                logger.debug(
                    event="sample_corpus_per_day",
                    context={
                        "day": day,
                        "session": session,
                        "sampled_article_count": len(article_samples),
                    },
                )
    return article_samples


def sample_per_day_session(daily_sample_df: pd.DataFrame) -> set[str]:
    """
    Apply boilerplate removal, substring exclusion, and language-confidence filtering
    to a single session’s sample for one trading day.

    Parameters
    ----------
    daily_sample_df : pandas.DataFrame
        DataFrame with columns ['article_id', 'language_confidence', 'full_text'].

    Returns
    -------
    set[str]
        Set of article_ids that pass all gating filters.

    Notes
    -----
    - Removes articles starting with known noisy prefixes.
    - Removes articles containing known noisy substrings.
    - Requires language_confidence ≥ STRONG_ENGLISH_CONFIDENCE_THRESHOLD.
    """

    boilerplate_prefix_mask: pd.Series = daily_sample_df["full_text"].str.startswith(NOISY_PREFIXES)
    boilerplate_substring_mask: pd.Series = daily_sample_df["full_text"].str.contains(
        "|".join(NOISY_SUBSTRINGS)
    )
    strong_english_mask: pd.Series = (
        daily_sample_df["language_confidence"] >= STRONG_ENGLISH_CONFIDENCE_THRESHOLD
    )
    clean_articles_df: pd.DataFrame = daily_sample_df.loc[
        ~boilerplate_prefix_mask & ~boilerplate_substring_mask & strong_english_mask
    ]
    return set(clean_articles_df["article_id"].tolist())


def batch_canonicalize_and_tokenize_corpus(
    filtered_sample_df: pd.DataFrame,
    logger: InfraLogger,
    real_run: bool = False,
    resume_index: int = 0,
) -> None:
    """
    Process the cleaned sample in CHUNK_SIZE batches, canonicalize tokens, and
    write tokenized parquet chunks to disk.

    Parameters
    ----------
    filtered_sample_df : pandas.DataFrame
        DataFrame of article_id + full_text for all sampled articles.
    logger : InfraLogger
        Structured logger for progress/debug statements.
    real_run : bool
        If False, the function no-ops and writes no parquet.
    resume_index : int
        Starting chunk index when resuming a partial run.

    Returns
    -------
    None

    Notes
    -----
    - Writes parquet files named tokenized_corpus_chunk_<k>.parquet.
    - Stateless across chunks; all computation occurs per-batch.
    """

    if not real_run:
        return
    TOKENIZED_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    nltk.download("stopwords", quiet=True)
    start_offset: int = resume_index * CHUNK_SIZE
    for start in range(start_offset, len(filtered_sample_df), CHUNK_SIZE):
        end: int = min(start + CHUNK_SIZE, len(filtered_sample_df))
        corpus_chunk: pd.DataFrame = filtered_sample_df.iloc[start:end].copy()
        tokenized_corpus_chunk: pd.DataFrame = canonicalize_and_tokenize_chunk(corpus_chunk, logger)
        tokenized_corpus_chunk.to_parquet(
            f"{TOKENIZED_PARQUET_DIR}/tokenized_corpus_chunk_{start // CHUNK_SIZE + 1}.parquet",
            index=False,
        )
        del corpus_chunk, tokenized_corpus_chunk


def canonicalize_and_tokenize_chunk(
    filtered_sample_df: pd.DataFrame, logger: InfraLogger
) -> pd.DataFrame:
    """
    Apply full normalization, token-type tagging, numeric canonicalization,
    alphanumeric cleanup, stop-word removal, and stemming to a corpus chunk.

    Parameters
    ----------
    filtered_sample_df : pandas.DataFrame
        Must contain ['article_id', 'full_text'].
    logger : InfraLogger
        Logger for stage-level debugging.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing ['article_id', 'stemmed_tokens'].

    Notes
    -----
    - Implements a deterministic, multi-stage cleaning pipeline.
    - numeric tokens become __NUM__/__MIL__/__BIL__.
    - Removes stop-words after stemming.
    """
    logger.debug(event="canonicalize_and_tokenize_corpus")
    filtered_sample_df = normalize_and_tokenize_sample(filtered_sample_df)
    logger.debug(event="normalized_and_tokenized_sample")
    filtered_sample_df = extract_token_types(filtered_sample_df)
    logger.debug(event="extracted_token_types")
    filtered_sample_df = canonicalize_numerical_tokens(filtered_sample_df)
    logger.debug(event="canonicalized_numerical_tokens")
    filtered_sample_df = canonicalize_alpha_numeric_tokens(filtered_sample_df)
    logger.debug(event="canonicalized_alpha_numeric_tokens")
    filtered_sample_df = stem_and_remove_stop_words(filtered_sample_df)
    logger.debug(event="stemmed_and_removed_stop_words")
    filtered_sample_df = clean_corpus(filtered_sample_df)
    logger.debug(event="cleaned_corpus")
    return filtered_sample_df[["article_id", "stemmed_tokens"]]


def normalize_and_tokenize_sample(filtered_sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize text, strip non-alphanumeric ASCII, split into raw tokens,
    and explode rows so that each row holds one token.

    Parameters
    ----------
    filtered_sample_df : pandas.DataFrame
        Must contain a 'full_text' column.

    Returns
    -------
    pandas.DataFrame
        Contains normalized 'full_text_normalized', tokenized 'raw_tokens', exploded
        to one token per row.

    Notes
    -----
    - All punctuation removed before token splitting.
    - Exploding ensures uniform token-level processing downstream.
    """

    filtered_sample_df["full_text_normalized"] = (
        filtered_sample_df["full_text"].str.replace(r"[^A-Z0-9\s]", " ", regex=True).str.strip()
    )
    filtered_sample_df["raw_tokens"] = filtered_sample_df["full_text_normalized"].str.split()
    filtered_sample_df = filtered_sample_df.explode("raw_tokens")
    return filtered_sample_df


def extract_token_types(filtered_sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each raw token into alphabetic, numeric, or alphanumeric classes.

    Parameters
    ----------
    filtered_sample_df : pandas.DataFrame
        Must contain a 'raw_tokens' column.

    Returns
    -------
    pandas.DataFrame
        Same rows with added 'token_key' and 'token_type'.

    Notes
    -----
    - token_key ∈ {1, 2, 3} encodes alpha/numeric/alphanumeric.
    """

    alphabetic_mask: pd.Series = filtered_sample_df["raw_tokens"].str.isalpha()
    numeric_mask: pd.Series = filtered_sample_df["raw_tokens"].str.isnumeric()
    conditions: List[pd.Series] = [
        alphabetic_mask,
        numeric_mask,
        (~alphabetic_mask & ~numeric_mask),
    ]
    numerical_keys: List[int] = [1, 2, 3]
    filtered_sample_df["token_key"] = np.select(conditions, numerical_keys)
    token_types = {
        1: "alphabetic",
        2: "numeric",
        3: "alphanumeric",
    }
    filtered_sample_df["token_type"] = filtered_sample_df["token_key"].map(token_types)
    return filtered_sample_df


def canonicalize_numerical_tokens(filtered_sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign numeric tokens to canonical magnitude buckets (__BIL__, __MIL__, __NUM__).

    Parameters
    ----------
    filtered_sample_df : pandas.DataFrame
        Must contain 'token_type' and 'raw_tokens'.

    Returns
    -------
    pandas.DataFrame
        Adds 'numeric_value', 'numerical_token_key', and 'canonical_numerical_token'.

    Notes
    -----
    - numeric_value is coerced with pandas.to_numeric(errors="coerce").
    - Magnitude thresholds come from config: BIL, MIL.
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
        1: "__BIL__",
        2: "__MIL__",
        3: "__NUM__",
    }
    filtered_sample_df["canonical_numerical_token"] = filtered_sample_df["numerical_token_key"].map(
        numerical_token_types
    )

    return filtered_sample_df


def canonicalize_alpha_numeric_tokens(filtered_sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip digits out of alphanumeric tokens to yield pure alphabetic forms.

    Parameters
    ----------
    filtered_sample_df : pandas.DataFrame
        Must contain 'raw_tokens' and 'token_type'.

    Returns
    -------
    pandas.DataFrame
        Same rows; alphanumeric tokens modified in-place.

    Notes
    -----
    - Only applies to tokens where token_type == 'alphanumeric' but not strictly
      'alphabetic' or 'numeric'.
    """

    alphanumeric_token_mask: pd.Series = filtered_sample_df["token_type"] == "alphanumeric"
    filtered_sample_df["raw_tokens"] = np.where(
        alphanumeric_token_mask,
        filtered_sample_df["raw_tokens"].str.replace(r"[0-9]", "", regex=True),
        filtered_sample_df["raw_tokens"],
    )
    return filtered_sample_df


def stem_and_remove_stop_words(filtered_sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stem all alphabetic tokens using SnowballStemmer and remove English stop-words
    (after stemming).

    Parameters
    ----------
    filtered_sample_df : pandas.DataFrame
        Must contain 'raw_tokens', 'token_type', and canonical numeric tokens.

    Returns
    -------
    pandas.DataFrame
        Rows with stop-words removed, with new column 'stemmed_tokens'.

    Notes
    -----
    - Numeric tokens bypass stemming and use canonical_numerical_token.
    - Stop-words are stemmed before comparison, ensuring consistency.
    """

    stemmer: nltk.SnowballStemmer = nltk.SnowballStemmer("english")
    filtered_sample_df["stemmed_tokens"] = np.where(
        filtered_sample_df["token_type"] == "numeric",
        filtered_sample_df["canonical_numerical_token"],
        filtered_sample_df["raw_tokens"].str.lower().apply(stemmer.stem),
    )
    stemmed_stop_words_set: set[str] = set(
        stemmer.stem(word) for word in nltk.corpus.stopwords.words("english")
    )
    filtered_sample_df = filtered_sample_df[
        ~filtered_sample_df["stemmed_tokens"].isin(stemmed_stop_words_set)
    ]
    return filtered_sample_df


def clean_corpus(
    filtered_sample_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply final cleaning: remove too-short tokens and any tokens containing digits.

    Parameters
    ----------
    filtered_sample_df : pandas.DataFrame
        Must contain 'stemmed_tokens'.

    Returns
    -------
    pandas.DataFrame
        Filtered view restricted to tokens satisfying length/digit constraints.

    Notes
    -----
    - Minimal length comes from MINIMAL_CHAR_COUNT_PER_TOKEN.
    """

    stems_longer_then_minimal: pd.Series = (
        filtered_sample_df["stemmed_tokens"].str.len() > MINIMAL_CHAR_COUNT_PER_TOKEN
    )
    stems_with_digits: pd.Series = filtered_sample_df["stemmed_tokens"].str.contains(
        r"\d", regex=True
    )
    return filtered_sample_df[stems_longer_then_minimal & ~stems_with_digits]


def extract_token_distributions() -> FrequencyCounters:
    """
    Scan tokenized parquet chunks and build token-level frequency counters for the
    (unpruned) vocabulary.

    Parameters
    ----------
    None

    Returns
    -------
    FrequencyCounters
        FrequencyCounters instance with:
        - token_frequency_counter populated with global term counts, and
        - token_document_counter populated with document-frequency counts.
        Document-level counters remain None and are intended to be filled in a
        subsequent pass after vocabulary pruning.

    Notes
    -----
    - Each parquet file in `tokenized_parquet` must contain ['article_id',
      'stemmed_tokens'].
    - This function is intentionally lightweight: it only builds token-level
      statistics. Document-level counts are derived later by
      summarize_and_filter_vocabulary / extract_per_document_per_term_counters.
    """

    token_frequency_counter: Counter[str] = Counter()
    token_document_counter: Counter[str] = Counter()

    parquet_dir: Path = TOKENIZED_PARQUET_DIR
    parquet_paths = sorted(parquet_dir.glob("tokenized_corpus_chunk_*.parquet"))

    for path in parquet_paths:
        df = pd.read_parquet(path)
        token_frequency_counter.update(df["stemmed_tokens"].to_numpy())
        unique_pairs = df.drop_duplicates(subset=["article_id", "stemmed_tokens"])
        token_document_counter.update(unique_pairs["stemmed_tokens"].to_numpy())
        del df, unique_pairs
    return FrequencyCounters(token_frequency_counter, token_document_counter)


def summarize_and_filter_vocabulary(
    frequency_counters: FrequencyCounters,
    min_doc_freq: int = 25,
) -> FrequencyCounters:
    """
    Summarize the raw vocabulary, apply a minimum document-frequency filter, print
    diagnostics, and compute document-level statistics for the kept tokens.

    Parameters
    ----------
    frequency_counters : FrequencyCounters
        Frequency counters built from the unpruned token universe. On return, this
        instance is updated to reflect the pruned vocabulary and document-level
        counts.
    min_doc_freq : int, optional
        Minimum document frequency required for a token to be retained in the
        active vocabulary.

    Returns
    -------
    FrequencyCounters
        The same FrequencyCounters instance, updated so that:
        - token_frequency_counter and token_document_counter correspond to the
          pruned vocabulary, and
        - document_global_counter, document_unique_counter, and
          token_within_document_counter are computed with respect to
          that pruned vocabulary.

    Notes
    -----
    - Prints a small diagnostic report to stdout:
    - top 20 tokens by global term count, and
    - before/after vocabulary sizes and removal fraction.
    - This function is responsible for defining the active vocabulary for a given
      corpus_version; downstream ingestion should treat its output as canonical.
    """

    frequency_counter: Counter[str] = frequency_counters.token_frequency_counter
    document_counter: Counter[str] = frequency_counters.token_document_counter
    all_tokens: set[str] = set(frequency_counter.keys()) | set(document_counter.keys())
    vocab_size_before: int = len(all_tokens)

    kept_tokens: set[str] = {t for t in all_tokens if document_counter.get(t, 0) >= min_doc_freq}
    vocab_size_after: int = len(kept_tokens)

    frequency_counters.token_frequency_counter = Counter(
        {t: frequency_counter[t] for t in kept_tokens if t in frequency_counter}
    )
    frequency_counters.token_document_counter = Counter(
        {t: document_counter[t] for t in kept_tokens if t in document_counter}
    )

    frequency_counters = extract_per_document_per_term_counters(
        frequency_counters, kept_tokens
    )

    print("Top 20 tokens by global term count")
    print("-" * 60)
    print(f"{'rank':>4}  {'token':<20} {'term_count':>12} {'doc_freq':>12}")
    for i, (token, count) in enumerate(
        frequency_counters.token_frequency_counter.most_common(20), start=1
    ):
        df = document_counter.get(token, 0)
        print(f"{i:>4}  {token:<20} {count:>12} {df:>12}")

    removed = vocab_size_before - vocab_size_after
    frac_removed = removed / vocab_size_before if vocab_size_before else 0.0

    print()
    print(f"Vocabulary summary (min document frequency = {min_doc_freq})")
    print("-" * 60)
    print(f"{'vocab_size_before':<25}{vocab_size_before:>10}")
    print(f"{'vocab_size_after':<25}{vocab_size_after:>10}")
    print(f"{'tokens_removed':<25}{removed:>10}")
    print(f"{'fraction_removed':<25}{frac_removed:>10.4f}")

    return frequency_counters


def extract_per_document_per_term_counters(
    frequency_counters: FrequencyCounters, vocab: set[str]
) -> FrequencyCounters:
    """
    Recompute per-document and per-(document, token) term statistics from parquet
    for a given pruned vocabulary.

    Parameters
    ----------
    frequency_counters : FrequencyCounters
        Frequency counters holding the token-level view. This instance is mutated
        in place to populate document_global_counter, document_unique_counter, and
        token_within_document_counter for the provided vocab.
    vocab : set[str]
        Active vocabulary tokens. Only rows whose stemmed_tokens belong to this
        set are counted.

    Returns
    -------
    FrequencyCounters
        The same FrequencyCounters instance, with:
        - document_global_counter populated as article_id → total token count
          restricted to vocab,
        - document_unique_counter populated as article_id → number of unique tokens
          restricted to vocab, and
        - token_within_document_counter populated as (article_id, token) → term_count
          for tokens in vocab.

    Notes
    -----
    - Re-reads all tokenized parquet chunks from the `tokenized_parquet` directory.
    - All aggregation is vectorized via pandas groupby; the Counter updates only
      materialize the grouped statistics.
    - This function assumes that the parquet files expose columns
      ['article_id', 'stemmed_tokens'] that are already cleaned and canonicalized.
    """

    document_global_counter: Counter[str] = Counter()
    document_unique_counter: Counter[str] = Counter()
    token_within_document_counter: Counter[tuple[str, str]] = Counter()

    parquet_dir: Path = TOKENIZED_PARQUET_DIR
    parquet_paths = sorted(parquet_dir.glob("tokenized_corpus_chunk_*.parquet"))
    for path in parquet_paths:
        df = pd.read_parquet(path)
        df = df[df["stemmed_tokens"].isin(vocab)]
        article_grouped = df.groupby("article_id")["stemmed_tokens"].agg(
            global_count="count",
            unique_count="nunique",
        )
        article_token_grouped = (
            df.groupby(["article_id", "stemmed_tokens"]).size().reset_index(name="term_count")
        )
        document_global_counter.update(
            dict(zip(article_grouped.index, article_grouped["global_count"]))
        )
        document_unique_counter.update(
            dict(zip(article_grouped.index, article_grouped["unique_count"]))
        )
        token_within_document_counter.update(
            {
                (row["article_id"], row["stemmed_tokens"]): row["term_count"]
                for _, row in article_token_grouped.iterrows()
            }
        )
        del df, article_grouped, article_token_grouped
    frequency_counters.document_global_counter = document_global_counter
    frequency_counters.document_unique_counter = document_unique_counter
    frequency_counters.token_within_document_counter = token_within_document_counter
    return frequency_counters


def delete_parquet_chunks(real_run: bool = False) -> None:
    """
    Delete all Parquet chunk files under TOKENIZED_PARQUET_DIR.

    Parameters
    ----------
    real_run : bool
        If False, the function no-ops and does not delete any files.

    Returns
    -------
    None
        All files matching "*.parquet" under `root` are deleted if they exist.

    Notes
    -----
    - Only files with the ".parquet" extension are removed; the directory
      itself is left in place.
    - Intended for local cleanup after running the LDA corpus preprocessing
      notebook.
    """
    if not real_run:
        return None
    root_path = Path(TOKENIZED_PARQUET_DIR)
    if not root_path.exists():
        print(f"[cleanup] Directory not found: {root_path}")
        return

    files_deleted = 0
    for parquet_file in root_path.glob("*.parquet"):
        parquet_file.unlink()
        files_deleted += 1

    print(f"[cleanup] Deleted {files_deleted} Parquet file(s) from {root_path}")
