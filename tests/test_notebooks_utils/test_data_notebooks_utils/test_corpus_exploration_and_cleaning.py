"""
Purpose
-------
Exercise the branching and aggregation logic in the
`corpus_exploration_and_cleaning_utils` module. The tests focus on text
normalization, token classification, numeric canonicalization, vocabulary
pruning, and post-pruning document-level statistics.

Key behaviors
-------------
- Validate that noisy articles are filtered based on configured prefixes,
  substrings, and language-confidence thresholds.
- Verify that the multi-stage tokenization pipeline (normalize → classify →
  canonicalize → stem → clean) produces the expected token representations.
- Check that token-level frequency and document-frequency counters are
  computed correctly from tokenized parquet chunks.
- Assert that vocabulary pruning by minimum document frequency yields a
  consistent active vocabulary and aligned document-level statistics.

Conventions
-----------
- Tests use pytest and rely on temporary directories for writing parquet
  files; no real database connections are exercised.
- The current working directory is changed to a temporary path when testing
  functions that read from `tokenized_parquet/`.
- External configuration values (e.g., BIL, MIL, MINIMAL_CHAR_COUNT_PER_TOKEN)
  are imported from the module under test to avoid duplication.

Downstream usage
----------------
Run these tests with pytest as part of the CI pipeline to guard the
corpus-processing invariants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, cast

import pandas as pd
import pytest

# fmt: off
from notebooks_utils.data_notebooks_utils.corpus_exploration_and_cleaning_utils.\
    corpus_exploration_and_cleaning_utils import (
    FrequencyCounters,
)

from infra.logging.infra_logger import InfraLogger
from notebooks_utils.data_notebooks_utils.corpus_exploration_and_cleaning_utils import (
    corpus_exploration_and_cleaning_config as corpus_utils_config,
)
from notebooks_utils.data_notebooks_utils.corpus_exploration_and_cleaning_utils import (
    corpus_exploration_and_cleaning_utils as corpus_utils,
)

# fmt: on


class DummyInfraLogger:
    """
    Purpose
    -------
    Provide a minimal stand-in for `InfraLogger` that satisfies the logging
    interface expected by the corpus-processing functions without performing
    any I/O.

    Key behaviors
    -------------
    - Exposes a `debug` method that accepts arbitrary positional and keyword
      arguments and discards them.
    - Can be passed anywhere an `InfraLogger` instance is required in the
      corpus utils module.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Notes
    -----
    - This class is intentionally no-op; it only exists to decouple tests
      from the concrete logging implementation.
    """

    def debug(self, *args: object, **kwargs: object) -> None:
        """
        Accept arbitrary arguments and perform no operation.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded from the caller.
        **kwargs : object
            Keyword arguments forwarded from the caller.

        Returns
        -------
        None
            No value is returned; the call is intentionally a no-op.

        Notes
        -----
        - This method mirrors the signature of a typical logger `debug`
          method so it can be used as a drop-in replacement in tests.
        """
        return None


@pytest.fixture
def dummy_logger() -> DummyInfraLogger:
    """
    Construct a dummy logger suitable for use in corpus-processing tests.

    Parameters
    ----------
    None

    Returns
    -------
    DummyInfraLogger
        A logger instance whose `debug` method ignores all input.

    Notes
    -----
    - This fixture avoids importing the real `InfraLogger` into the test
      module and keeps tests focused on the data-processing logic.
    """
    return DummyInfraLogger()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Build a minimal daily sample DataFrame for testing article filtering
    logic in `sample_per_day_session`.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ['article_id', 'language_confidence',
        'full_text'] and a mix of clean and noisy rows.

    Notes
    -----
    - The prefixes and substrings used to construct noisy examples are
      pulled from the module-level configuration so the test remains
      robust to config changes.
    """
    noisy_prefix: str = corpus_utils_config.NOISY_PREFIXES[0]
    noisy_substring: str = sorted(corpus_utils_config.NOISY_SUBSTRINGS)[0]
    threshold: float = corpus_utils_config.STRONG_ENGLISH_CONFIDENCE_THRESHOLD

    data = {
        "article_id": [
            "clean_article",
            "noisy_prefix_article",
            "noisy_substring_article",
            "low_conf_article",
        ],
        "language_confidence": [
            threshold,
            threshold,
            threshold,
            threshold - 0.1,
        ],
        "full_text": [
            "THIS IS A CLEAN ARTICLE BODY.",
            f"{noisy_prefix} THIS ARTICLE SHOULD BE REMOVED BY PREFIX.",
            f"THIS ARTICLE CONTAINS A {noisy_substring} THAT SHOULD BE FILTERED.",
            "BODY IS IN ENGLISH BUT HAS LOW CONFIDENCE SCORE.",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def tmp_tokenized_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Create a small synthetic tokenized corpus written to a temporary
    `tokenized_parquet` directory and adjust the working directory so the
    corpus utils module can discover it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory unique to this test run.
    monkeypatch : pytest.MonkeyPatch
        Utility for manipulating environment and working directory state.

    Returns
    -------
    pathlib.Path
        Path to the `tokenized_parquet` directory containing the generated
        parquet files.

    Notes
    -----
    - The directory structure and filenames mirror those used by the
      production pipeline (`tokenized_corpus_chunk_<k>.parquet`).
    - The token distribution is chosen to make vocabulary pruning behavior
      easy to reason about in downstream tests.
    """
    # Change CWD so corpus_utils.Path("tokenized_parquet") points inside tmp_path
    monkeypatch.chdir(tmp_path)

    tokenized_dir: Path = tmp_path / "tokenized_parquet"
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    # Construct a small corpus:
    # - doc1: tokens ["keep", "drop"]
    # - doc2: tokens ["keep"]
    # - doc3: tokens ["keep", "keep"]
    # This yields:
    #   global counts: keep=4, drop=1
    #   doc freq: keep=3 docs, drop=1 doc
    df_chunk_1 = pd.DataFrame(
        {
            "article_id": ["doc1", "doc1", "doc2"],
            "stemmed_tokens": ["keep", "drop", "keep"],
        }
    )
    df_chunk_2 = pd.DataFrame(
        {
            "article_id": ["doc3", "doc3"],
            "stemmed_tokens": ["keep", "keep"],
        }
    )

    df_chunk_1.to_parquet(
        tokenized_dir / "tokenized_corpus_chunk_1.parquet",
        index=False,
    )
    df_chunk_2.to_parquet(
        tokenized_dir / "tokenized_corpus_chunk_2.parquet",
        index=False,
    )

    return tokenized_dir


def test_sample_per_day_session_filters_boilerplate_and_language(
    sample_df: pd.DataFrame,
) -> None:
    """
    Verify that `sample_per_day_session` drops rows with noisy prefixes,
    noisy substrings, or insufficient language confidence while retaining
    clean rows.

    Parameters
    ----------
    simple_daily_sample_df : pandas.DataFrame
        Synthetic session-level sample containing both clean and noisy
        articles.

    Returns
    -------
    None
        The test passes if only the clean article_id is retained.

    Notes
    -----
    - This test relies on the configured noisy prefixes and substrings but
      does not assume specific literal values.
    """
    result_ids: Set[str] = corpus_utils.sample_per_day_session(sample_df)
    assert result_ids == {"clean_article"}


def test_normalize_and_tokenize_sample_and_explodes() -> None:
    """
    Ensure that `normalize_and_tokenize_sample` strips
    non-alphanumeric characters, splits into tokens, and explodes the
    DataFrame to one token per row.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the normalized text and resulting tokens match
        expectations.

    Notes
    -----
    - This test intentionally uses punctuation and digits to confirm that
      non-alphanumeric characters are removed before splitting.
    """
    df = pd.DataFrame({"full_text": ["HELLO, WORLD! 123"]})
    result: pd.DataFrame = corpus_utils.normalize_and_tokenize_sample(df.copy())

    # All normalized entries should be identical.
    unique_normalized = set(result["full_text_normalized"].unique().tolist())
    assert unique_normalized == {"HELLO  WORLD  123"}

    # The exploded tokens should be the three alphanumeric segments.
    tokens: Set[str] = set(result["raw_tokens"].tolist())
    assert tokens == {"HELLO", "WORLD", "123"}


def test_extract_token_types_classifies_alpha_numeric_and_alphanumeric() -> None:
    """
    Confirm that `extract_token_types` correctly classifies tokens into
    alphabetic, numeric, and alphanumeric categories.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if token_type and token_key are assigned as
        expected for each input token.
    """
    df = pd.DataFrame({"raw_tokens": ["ABC", "123", "A1B2"]})
    result: pd.DataFrame = corpus_utils.extract_token_types(df.copy())

    types = dict(zip(result["raw_tokens"], result["token_type"]))
    keys = dict(zip(result["raw_tokens"], result["token_key"]))

    assert types["ABC"] == "alphabetic"
    assert types["123"] == "numeric"
    assert types["A1B2"] == "alphanumeric"

    assert keys["ABC"] == 1
    assert keys["123"] == 2
    assert keys["A1B2"] == 3


def test_canonicalize_numerical_tokens_assigns_magnitude_buckets() -> None:
    """
    Validate that `canonicalize_numerical_tokens` places numeric tokens into
    the correct magnitude buckets (__BIL__, __MIL__, __NUM__).

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the canonical_numerical_token values match the
        configured thresholds.

    Notes
    -----
    - Uses the BIL and MIL values imported from the module under test so
      the thresholds remain consistent with production configuration.
    """
    bil: float = float(corpus_utils.BIL)
    mil: float = float(corpus_utils.MIL)

    df = pd.DataFrame(
        {
            "raw_tokens": [str(int(bil * 2)), str(int(mil * 2)), "5"],
            "token_type": ["numeric", "numeric", "numeric"],
        }
    )

    result: pd.DataFrame = corpus_utils.canonicalize_numerical_tokens(df.copy())
    mapping = dict(zip(result["raw_tokens"], result["canonical_numerical_token"]))

    assert mapping[str(int(bil * 2))] == "__BIL__"
    assert mapping[str(int(mil * 2))] == "__MIL__"
    assert mapping["5"] == "__NUM__"


def test_canonicalize_alpha_numeric_tokens_strips_digits_only_for_alphanumeric() -> None:
    """
    Check that `canonicalize_alpha_numeric_tokens` strips digits from
    alphanumeric tokens while leaving purely alphabetic and purely numeric
    tokens unchanged.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if only the alphanumeric token loses its digits.
    """
    df = pd.DataFrame(
        {
            "raw_tokens": ["A1B2", "ABC", "123"],
            "token_type": ["alphanumeric", "alphabetic", "numeric"],
        }
    )

    result: pd.DataFrame = corpus_utils.canonicalize_alpha_numeric_tokens(df.copy())
    mapping = dict(zip(result["token_type"], result["raw_tokens"]))

    assert mapping["alphanumeric"] == "AB"
    assert mapping["alphabetic"] == "ABC"
    assert mapping["numeric"] == "123"


def test_stem_and_remove_stop_words_applies_stemming_and_drops_stopwords() -> None:
    """
    Ensure that `stem_and_remove_stop_words` stems alphabetic tokens, uses
    canonical numeric tokens as-is, and removes English stop words.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if numeric tokens are preserved with their canonical
        form, common stop words are removed, and non-stopword tokens are
        stemmed.
    """
    df = pd.DataFrame(
        {
            "raw_tokens": ["100", "the", "markets"],
            "token_type": ["numeric", "alphabetic", "alphabetic"],
            "canonical_numerical_token": ["__NUM__", None, None],
        }
    )

    result: pd.DataFrame = corpus_utils.stem_and_remove_stop_words(df.copy())
    tokens: List[str] = result["stemmed_tokens"].tolist()

    # Numeric token should be preserved via canonical_numerical_token.
    assert "__NUM__" in tokens
    # "the" should be removed as a stop word.
    assert "the" not in tokens
    # "markets" should be stemmed to something short like "market".
    assert any(tok.startswith("market") for tok in tokens)


def test_clean_corpus_filters_by_min_length_and_digits() -> None:
    """
    Verify that `clean_corpus` removes tokens that are too short or contain
    digits, and retains long-enough, digit-free tokens.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if only the long, digit-free token remains.
    """
    min_len: int = int(corpus_utils.MINIMAL_CHAR_COUNT_PER_TOKEN)

    stems = [
        "a" * (max(min_len - 1, 0)),  # too short
        "b" * min_len,  # boundary (== MINIMAL_CHAR_COUNT_PER_TOKEN)
        "c" * (min_len + 1) + "1",  # long but contains digit
        "d" * (min_len + 1),  # valid: long and no digits
    ]

    df = pd.DataFrame({"stemmed_tokens": stems})
    result: pd.DataFrame = corpus_utils.clean_corpus(df.copy())

    kept: List[str] = result["stemmed_tokens"].tolist()
    assert kept == ["d" * (min_len + 1)]


def test_canonicalize_and_tokenize_chunk_produces_article_stemmed_tokens(
    dummy_logger: DummyInfraLogger,
) -> None:
    """
    Confirm that `canonicalize_and_tokenize_chunk` runs the full text-cleaning
    pipeline and returns a DataFrame with article_id and stemmed_tokens.

    Parameters
    ----------
    dummy_logger : DummyInfraLogger
        A logger instance whose debug calls are ignored.

    Returns
    -------
    None
        The test passes if the output has the expected columns and at least
        one token row per input article.
    """
    df = pd.DataFrame(
        {
            "article_id": ["art1"],
            "full_text": ["Company profits increased by 100 million dollars."],
        }
    )
    logger = cast(InfraLogger, dummy_logger)  # for type-checker compatibility
    result: pd.DataFrame = corpus_utils.canonicalize_and_tokenize_chunk(df.copy(), logger)
    assert set(result.columns) == {"article_id", "stemmed_tokens"}
    assert (result["article_id"] == "art1").all()
    assert len(result) > 0


def test_extract_token_distributions_builds_token_level_counters(
    tmp_tokenized_corpus: Path,
) -> None:
    """
    Check that `extract_token_distributions` reads the tokenized parquet
    corpus and builds correct global and document-frequency counters.

    Parameters
    ----------
    tmp_tokenized_corpus : pathlib.Path
        Path to the temporary `tokenized_parquet` directory created for this
        test run.

    Returns
    -------
    None
        The test passes if token_frequency_counter and token_document_counter
        match the known synthetic corpus.
    """
    freq_counters: FrequencyCounters = corpus_utils.extract_token_distributions()

    tf: Dict[str, int] = dict(freq_counters.token_frequency_counter)
    df: Dict[str, int] = dict(freq_counters.token_document_counter)

    # From the synthetic corpus:
    #   global counts: keep=4, drop=1
    #   doc freq: keep=3 docs, drop=1 doc
    assert tf["keep"] == 4
    assert tf["drop"] == 1

    assert df["keep"] == 3
    assert df["drop"] == 1


def test_summarize_and_filter_vocabulary_prunes_and_aligns_doc_stats(
    tmp_tokenized_corpus: Path,
) -> None:
    """
    Validate that `summarize_and_filter_vocabulary` prunes tokens below the
    minimum document frequency and recomputes document-level statistics for
    the retained vocabulary.

    Parameters
    ----------
    tmp_tokenized_corpus : pathlib.Path
        Path to the temporary `tokenized_parquet` directory created for this
        test run.

    Returns
    -------
    None
        The test passes if:
        - tokens below the min_doc_freq threshold are removed from the active
          vocabulary, and
        - the document-level token counts correspond to the pruned vocabulary.

    Notes
    -----
    - Uses min_doc_freq=2 so that "drop" (doc_freq=1) is removed and "keep"
      (doc_freq=3) is retained.
    """
    base_counters: FrequencyCounters = corpus_utils.extract_token_distributions()
    min_doc_freq: int = 2

    updated_counters: FrequencyCounters = corpus_utils.summarize_and_filter_vocabulary(
        base_counters,
        min_doc_freq=min_doc_freq,
    )

    # After pruning, only "keep" should remain in the token-level counters.
    tf_pruned: Dict[str, int] = dict(updated_counters.token_frequency_counter)
    df_pruned: Dict[str, int] = dict(updated_counters.token_document_counter)

    assert set(tf_pruned.keys()) == {"keep"}
    assert set(df_pruned.keys()) == {"keep"}
    assert tf_pruned["keep"] == 4
    assert df_pruned["keep"] == 3

    # Document-level counts should be restricted to the pruned vocabulary:
    # doc1: originally ["keep", "drop"] → after pruning, 1 token, 1 unique.
    # doc2: ["keep"] → 1 token, 1 unique.
    # doc3: ["keep", "keep"] → 2 tokens, 1 unique.
    assert updated_counters.document_global_counter is not None
    assert updated_counters.document_unique_counter is not None
    assert updated_counters.token_within_document_counter is not None

    dg = dict(updated_counters.document_global_counter)
    du = dict(updated_counters.document_unique_counter)
    twd = dict(updated_counters.token_within_document_counter)

    assert dg["doc1"] == 1
    assert dg["doc2"] == 1
    assert dg["doc3"] == 2

    assert du["doc1"] == 1
    assert du["doc2"] == 1
    assert du["doc3"] == 1

    # Per-(doc, token) term counts.
    assert twd[("doc1", "keep")] == 1
    assert twd[("doc2", "keep")] == 1
    assert twd[("doc3", "keep")] == 2


def test_delete_parquet_chunks_removes_all_parquet_files(
    tmp_tokenized_corpus: Path,
) -> None:
    """
    Verify that `delete_parquet_chunks` removes all Parquet files from the
    tokenized corpus directory while leaving the directory itself intact.

    Parameters
    ----------
    tmp_tokenized_corpus : pathlib.Path
        Temporary `tokenized_parquet` directory populated by the fixture.

    Returns
    -------
    None
        The test passes if:
        - at least one Parquet file exists before deletion,
        - no Parquet files remain after calling `delete_parquet_chunks` with
          `real_run=True`, and
        - the `tokenized_parquet` directory still exists.

    Notes
    -----
    - Relies on the `tmp_tokenized_corpus` fixture to create Parquet files
      under the current working directory so that the corpus utils module
      discovers them via its hard-coded path.
    """
    # Sanity check: files exist before deletion.
    parquet_before = list(tmp_tokenized_corpus.glob("*.parquet"))
    assert len(parquet_before) > 0

    # Invoke the cleanup helper.
    corpus_utils.delete_parquet_chunks(real_run=True)

    # After deletion, no Parquet files should remain, but the directory should.
    parquet_after = list(tmp_tokenized_corpus.glob("*.parquet"))
    assert not parquet_after
    assert tmp_tokenized_corpus.exists()
