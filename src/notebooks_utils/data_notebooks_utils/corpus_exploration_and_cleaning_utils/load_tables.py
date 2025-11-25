"""
Purpose
-------
Bridge the LDA corpus preprocessing layer with the PostgreSQL ingestion layer.
This module takes aggregated `FrequencyCounters` objects and materializes them
into the normalized database schema (`lda_documents`, `lda_vocabulary`,
`lda_document_terms`) for a specific `corpus_version`.

Key behaviors
-------------
- Generates parameterized INSERT statements for the three LDA tables.
- Streams batched rows into the target tables using `load_into_table`, avoiding
  materializing large intermediate DataFrames in memory.
- Derives `term_id` values by re-reading the freshly populated
  `lda_vocabulary` table and joining them back to token-level counters.

Conventions
-----------
- The `corpus_version` argument is treated as an immutable identifier for a
  given corpus snapshot; all inserts for a single run must use the same value.
- The `frequency_counters` instance is assumed to be fully populated and
  self-consistent:
  - `token_frequency_counter` and `token_document_counter` reflect the pruned
    active vocabulary, and
  - `document_global_counter`, `document_unique_counter`, and
    `token_within_document_counter` are defined over the same token set.
- All DB interactions are performed within a context manager returned by
  `connect_to_db`, which is responsible for connection lifecycle and
  transaction behavior.
- INSERT statements use `ON CONFLICT DO NOTHING`, so re-running ingestion for
  the same `(article_id, corpus_version)` or `(token, corpus_version)` is
  idempotent at the row level.

Downstream usage
----------------
Call `load_corpus_tables` after constructing a `FrequencyCounters` instance
from tokenized parquet and applying vocabulary pruning. Downstream modeling
code should treat the populated tables (`lda_documents`, `lda_vocabulary`,
`lda_document_terms`) as the canonical representation of the corpus for a
given `corpus_version`.
"""

import warnings
from typing import Any, Iterable, cast

import pandas as pd

from infra.logging.infra_logger import InfraLogger
from infra.utils.db_utils import connect_to_db, load_into_table

# fmt: off
from notebooks_utils.data_notebooks_utils.corpus_exploration_and_cleaning_utils.\
    corpus_exploration_and_cleaning_utils import FrequencyCounters

# fmt: on


def load_corpus_tables(
    frequency_counters: FrequencyCounters,
    logger: InfraLogger,
    corpus_version: int = 1,
    real_run: bool = False,
) -> None:
    """
    Load LDA corpus statistics into the normalized DB tables for a given corpus_version.

    Parameters
    ----------
    frequency_counters : FrequencyCounters
        Aggregated token- and document-level counters for the active vocabulary.
        This object is expected to have:
        - `token_frequency_counter` and `token_document_counter` populated for
          all vocabulary tokens, and
        - `document_global_counter`, `document_unique_counter`, and
          `token_within_document_counter` populated for all articles.
    corpus_version : int, optional
        Integer identifier for the corpus snapshot being ingested. The same
        value is written to `lda_documents`, `lda_vocabulary`, and
        `lda_document_terms`.
    logger : InfraLogger
        Logger instance for recording progress and status messages.
    real_run : bool, optional
        If False, the function returns immediately without opening a database
        connection or performing any inserts. When True, the function executes
        all INSERT operations.

    Returns
    -------
    None
        Returns None; success is indicated by the absence of raised exceptions.

    Raises
    ------
    Exception
        Any errors raised by `connect_to_db`, `load_into_table`, or
        `pandas.read_sql` are propagated to the caller (e.g., connectivity
        issues, schema mismatches, or constraint violations).

    Notes
    -----
    - The ingestion order is:
      1. Insert into `lda_documents`.
      2. Insert into `lda_vocabulary`.
      3. Read back `lda_vocabulary` for the given `corpus_version` to obtain
         the mapping token → term_id.
      4. Insert into `lda_document_terms` using that mapping.
    - All INSERT statements use `ON CONFLICT DO NOTHING`, making ingestion
      idempotent with respect to duplicate rows.
    - The function is intentionally streaming-oriented: rows are provided as
      generators to `load_into_table` to avoid loading large result sets into
      memory.
    """
    if not real_run:
        return None
    lda_document_query: str = generate_lda_document_query()
    lda_vocabulary_query: str = generate_lda_vocabulary_query()
    lda_document_row_generator: Iterable[tuple] = create_lda_documents_row_generator(
        frequency_counters, corpus_version
    )
    lda_vocabulary_row_generator: Iterable[tuple] = create_lda_vocabulary_row_generator(
        frequency_counters, corpus_version
    )
    with connect_to_db() as conn:
        logger.debug("loading_lda_document_table")
        load_into_table(conn, lda_document_row_generator, lda_document_query)
        logger.debug("loading_lda_vocabulary_table")
        load_into_table(conn, lda_vocabulary_row_generator, lda_vocabulary_query)
        logger.debug("finished_loading_lda_vocabulary_table")

        # Reuse the existing DB-API connection with pandas and locally suppress
        # its advisory "use SQLAlchemy" warning; this call is still fully supported.
        conn_for_typing: Any = cast(Any, conn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            vocab_df: pd.DataFrame = pd.read_sql(
                "SELECT term_id, token FROM lda_vocabulary WHERE corpus_version = %s",
                conn_for_typing,
                params=(corpus_version,),
            )
        logger.debug("fetched_lda_vocabulary_from_db")
        token_to_term_id: dict[str, int] = dict(zip(vocab_df["token"], vocab_df["term_id"]))
        lda_document_terms_query: str = generate_lda_document_terms_query()
        lda_document_terms_row_generator: Iterable[tuple] = create_lda_document_terms_row_generator(
            frequency_counters, token_to_term_id, corpus_version
        )
        logger.debug("loading_lda_document_terms_table")
        load_into_table(conn, lda_document_terms_row_generator, lda_document_terms_query)


def generate_lda_document_query() -> str:
    """
    Build the parametrized INSERT statement used to populate `lda_documents`.

    Parameters
    ----------
    None

    Returns
    -------
    str
        A SQL string with a VALUES placeholder (`VALUES %s`) suitable for use
        with `load_into_table` when inserting rows of
        (article_id, corpus_version, token_count, unique_token_count).

    Raises
    ------
    None

    Notes
    -----
    - The statement uses `ON CONFLICT DO NOTHING` to suppress duplicate-key
      errors when a document row already exists for a given `(article_id,
      corpus_version)`.
    """

    return """
    INSERT INTO lda_documents(
        article_id,
        corpus_version,
        token_count,
        unique_token_count
    ) VALUES %s ON CONFLICT DO NOTHING;
    """


def generate_lda_vocabulary_query() -> str:
    """
    Build the parametrized INSERT statement used to populate `lda_vocabulary`.

    Parameters
    ----------
    None

    Returns
    -------
    str
        A SQL string with a VALUES placeholder (`VALUES %s`) suitable for use
        with `load_into_table` when inserting rows of
        (token, corpus_version, global_term_count, document_frequency).

    Raises
    ------
    None

    Notes
    -----
    - The statement uses `ON CONFLICT DO NOTHING` so re-ingesting the same
      (token, corpus_version) pair is a no-op rather than an error.
    """

    return """
    INSERT INTO lda_vocabulary(
        token,
        corpus_version,
        global_term_count,
        document_frequency
    ) VALUES %s ON CONFLICT DO NOTHING;
    """


def generate_lda_document_terms_query() -> str:
    """
    Build the parametrized INSERT statement used to populate `lda_document_terms`.

    Parameters
    ----------
    None

    Returns
    -------
    str
        A SQL string with a VALUES placeholder (`VALUES %s`) suitable for use
        with `load_into_table` when inserting rows of
        (article_id, corpus_version, term_id, term_count).

    Raises
    ------
    None

    Notes
    -----
    - The statement uses `ON CONFLICT DO NOTHING` to avoid raising if a given
      (article_id, corpus_version, term_id) row already exists.
    """

    return """
    INSERT INTO lda_document_terms(
        article_id,
        corpus_version,
        term_id,
        term_count
    ) VALUES %s ON CONFLICT DO NOTHING;
    """


def create_lda_documents_row_generator(
    frequency_counters: FrequencyCounters, corpus_version: int = 1
) -> Iterable[tuple]:
    """
    Stream `lda_documents` rows derived from document-level counters.

    Parameters
    ----------
    frequency_counters : FrequencyCounters
        Aggregated counters containing `document_global_counter` and
        `document_unique_counter` keyed by article_id. These counters are
        expected to be populated and aligned with the active vocabulary.
    corpus_version : int, optional
        Corpus identifier to attach to every generated row.

    Returns
    -------
    Iterable[tuple]
        A generator of tuples
        `(article_id, corpus_version, token_count, unique_token_count)` suitable
        for use with `load_into_table` and `generate_lda_document_query`.

    Raises
    ------
    KeyError
        If an article_id present in `document_global_counter` is missing from
        `document_unique_counter` (indicative of inconsistent counters).

    Notes
    -----
    - The iteration order follows the keys of `document_global_counter`; the
      DB schema does not assume any particular ordering.
    - `.get()` is used for lookups, but in a well-formed `FrequencyCounters`
      instance both document counters should contain the same article_id keys.
    """
    assert frequency_counters.document_global_counter is not None  # For type checker
    assert frequency_counters.document_unique_counter is not None  # For type checker
    for article_id in frequency_counters.document_global_counter.keys():
        yield (
            article_id,
            corpus_version,
            frequency_counters.document_global_counter.get(article_id),
            frequency_counters.document_unique_counter.get(article_id),
        )


def create_lda_vocabulary_row_generator(
    frequency_counters: FrequencyCounters, corpus_version: int = 1
) -> Iterable[tuple]:
    """
    Stream `lda_vocabulary` rows derived from token-level frequency counters.

    Parameters
    ----------
    frequency_counters : FrequencyCounters
        Aggregated counters containing:
        - `token_frequency_counter` with global term counts, and
        - `token_document_counter` with document-frequency counts.
        Both counters are expected to be defined on the same token set.
    corpus_version : int, optional
        Corpus identifier to attach to every generated row.

    Returns
    -------
    Iterable[tuple]
        A generator of tuples
        `(token, corpus_version, global_term_count, document_frequency)` suitable
        for use with `load_into_table` and `generate_lda_vocabulary_query`.

    Raises
    ------
    KeyError
        If a token present in `token_document_counter` is missing from
        `token_frequency_counter`, indicating inconsistent token-level counters.

    Notes
    -----
    - Iteration is over `token_document_counter.keys()` so that every retained
      vocabulary token appears at least once in some document.
    - The function assumes that `summarize_and_filter_vocabulary` has already
      pruned both counters to a consistent, final vocabulary.
    """
    for token in frequency_counters.token_document_counter.keys():
        yield (
            token,
            corpus_version,
            frequency_counters.token_frequency_counter[token],
            frequency_counters.token_document_counter[token],
        )


def create_lda_document_terms_row_generator(
    frequency_counters: FrequencyCounters, token_to_term_id: dict[str, int], corpus_version: int = 1
) -> Iterable[tuple]:
    """
    Stream `lda_document_terms` rows from per-(article, token) term counts.

    Parameters
    ----------
    frequency_counters : FrequencyCounters
        Aggregated counters containing `token_within_document_counter` keyed by
        `(article_id, token)` with integer term counts. This counter is assumed
        to be restricted to the active vocabulary.
    token_to_term_id : dict[str, int]
        Mapping from token string to `term_id` as stored in `lda_vocabulary`
        for the same `corpus_version`. All tokens in
        `token_within_document_counter` must appear in this mapping.
    corpus_version : int, optional
        Corpus identifier to attach to every generated row.

    Returns
    -------
    Iterable[tuple]
        A generator of tuples
        `(article_id, corpus_version, term_id, term_count)` suitable for use
        with `load_into_table` and `generate_lda_document_terms_query`.

    Raises
    ------
    KeyError
        If a token in `token_within_document_counter` is missing from
        `token_to_term_id`, indicating a mismatch between counters and the
        vocabulary table.

    Notes
    -----
    - This generator is typically consumed after `lda_vocabulary` has been
      populated and queried back to build `token_to_term_id`.
    - The iteration order over `token_within_document_counter.items()` is not
      semantically significant; the DB schema imposes no ordering.
    - The function assumes that `token_within_document_counter` is defined over the
      pruned vocabulary used in `lda_vocabulary`.
    """
    assert frequency_counters.token_within_document_counter is not None  # For type checker
    for (article_id, token), term_count in frequency_counters.token_within_document_counter.items():
        term_id: int = token_to_term_id[token]
        yield (
            article_id,
            corpus_version,
            term_id,
            term_count,
        )
