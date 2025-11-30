"""
Purpose
-------
Exercise the I/O wiring and metadata-assembly logic in
`coherence_measurement_utils` that is not covered by the lower-level
corpus-processing tests.

Key behaviors
-------------
- Validate that topic–word weight parquet files are discovered and read
  via the expected glob pattern.
- Check that top-N terms per topic are selected and ordered correctly
  from topic–word weight DataFrames.
- Verify that doc–topic exposure parquet files are annotated with the
  correct `run_id` and `corpus_version` metadata.
- Ensure that topic metadata DataFrames are constructed with consistent
  `run_id` patterns and that the default seed list is used when no
  explicit seeds are provided.

Conventions
-----------
- Tests rely on small synthetic DataFrames and fake Path classes rather
  than touching the real filesystem layout or database.
- The module under test is imported as
  `notebooks_utils.modeling_notebooks_utils.coherence_measurement_utils.coherence_measurement_utils`
  and aliased to `cmu`.
- Only the higher-level loading and metadata-assembly helpers are tested;
  corpus-cleaning functions are exercised in their own dedicated suite.

Downstream usage
----------------
Run these tests with pytest as part of the CI pipeline to guard the
contract around LDA artifact loading and topic metadata construction.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, List, Set

import pandas as pd
import pytest

from notebooks_utils.modeling_notebooks_utils.coherence_measurement_utils import (
    coherence_measurement_utils as cmu,
)


def test_extract_word_weight_dfs_uses_glob_and_read_parquet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `extract_word_weight_dfs` discovers topic-word weight parquet
    files via the expected glob pattern and returns one DataFrame per file.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest helper for patching the Path class and parquet reader within
        the module under test.

    Returns
    -------
    None
        The test passes if the glob pattern is correct, the fake parquet
        paths are read, and the number of returned DataFrames matches the
        number of synthetic files.

    Raises
    ------
    AssertionError
        If the glob pattern is not `K200_seed*.parquet`, if parquet files are
        not read as expected, or if the length of the returned list does not
        match the number of fake paths.

    Notes
    -----
    - Uses a fake Path implementation that ignores the real project layout
      and returns a controlled list of K=200 seed parquet files.
    """
    fake_paths: List[Path] = [
        Path("/fake/K200_seed42.parquet"),
        Path("/fake/K200_seed43.parquet"),
    ]

    class FakePath:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

        def resolve(self) -> "FakePath":
            return self

        @property
        def parents(self) -> List["FakePath"]:
            # Ensure parents[4] is defined and returns another FakePath.
            return [self, self, self, self, self]

        def __truediv__(self, other: str) -> "FakePath":
            # Chaining / just returns the same fake path object.
            return self

        def glob(self, pattern: str) -> List[Path]:
            assert pattern == "K200_seed*.parquet"
            return fake_paths

    read_calls: List[Path] = []

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        read_calls.append(path)
        return pd.DataFrame({"topic_id": [], "term": [], "weight": []})

    monkeypatch.setattr(cmu, "Path", FakePath)
    monkeypatch.setattr(cmu.pd, "read_parquet", fake_read_parquet)

    result: List[pd.DataFrame] = cmu.extract_word_weight_dfs()

    assert read_calls == fake_paths
    assert len(result) == len(fake_paths)


def test_extract_top_words_returns_top_n_terms_in_weight_order() -> None:
    """
    Ensure that `extract_top_words` selects the top-N highest-weight terms per
    topic and orders them by descending weight for each run.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if each topic's term list contains exactly N terms
        and they are sorted from highest to lowest weight.

    Raises
    ------
    AssertionError
        If the returned nested lists have incorrect lengths or if the term
        ordering does not match the expected weight ranking.

    Notes
    -----
    - Uses a single synthetic run with two topics and a small number of
      terms to make the expected ordering explicit.
    """
    df: pd.DataFrame = pd.DataFrame(
        {
            "topic_id": [0, 0, 0, 1, 1],
            "term": ["a", "b", "c", "d", "e"],
            "weight": [0.1, 0.4, 0.3, 0.2, 0.5],
        }
    )

    word_weight_dfs: List[pd.DataFrame] = [df]
    result: List[List[List[str]]] = cmu.extract_top_words(
        word_weight_dfs=word_weight_dfs,
        top_word_amount=2,
    )

    # One run in outer list.
    assert len(result) == 1
    run_topics: List[List[str]] = result[0]

    # Two topics in this run.
    assert len(run_topics) == 2

    # Topic 0: weights 0.4 (b), 0.3 (c), 0.1 (a) → top-2: ["b", "c"].
    assert run_topics[0] == ["b", "c"]

    # Topic 1: weights 0.5 (e), 0.2 (d) → top-2: ["e", "d"].
    assert run_topics[1] == ["e", "d"]


def test_extract_topic_exposure_dfs_sets_run_id_and_corpus_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Check that `extract_topic_exposure_dfs` loads doc-topic parquet files and
    annotates each DataFrame with the correct `run_id` and `corpus_version`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest helper for patching the Path class and parquet reader within
        the module under test.

    Returns
    -------
    None
        The test passes if the constructed `run_id` encodes the expected
        seeds and date window, and all `corpus_version` values match the
        input.

    Raises
    ------
    AssertionError
        If the `run_id` pattern is incorrect or if any `corpus_version`
        column contains unexpected values.

    Notes
    -----
    - Seed numbers are derived from filenames mirroring the production
      pattern `K200_seedXX.parquet`.
    """
    fake_paths: List[Path] = [
        Path("/fake/K200_seed42.parquet"),
        Path("/fake/K200_seed43.parquet"),
    ]

    class FakePath:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

        def resolve(self) -> "FakePath":
            return self

        @property
        def parents(self) -> List["FakePath"]:
            return [self, self, self, self, self]

        def __truediv__(self, other: str) -> "FakePath":
            return self

        def glob(self, pattern: str) -> List[Path]:
            assert pattern == "K200_seed*.parquet"
            return fake_paths

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "doc_index": [0],
                "article_id": ["A1"],
                "topic_id": [7],
                "topic_proportion": [0.5],
            }
        )

    monkeypatch.setattr(cmu, "Path", FakePath)
    monkeypatch.setattr(cmu.pd, "read_parquet", fake_read_parquet)

    sample_start: dt.date = dt.date(2016, 8, 1)
    sample_end: dt.date = dt.date(2016, 8, 2)
    corpus_version: int = 5

    dfs: List[pd.DataFrame] = cmu.extract_topic_exposure_dfs(
        sample_start=sample_start,
        sample_end=sample_end,
        corpus_version=corpus_version,
    )

    # One DataFrame per fake path.
    assert len(dfs) == len(fake_paths)

    run_ids: Set[str] = {df["run_id"].iloc[0] for df in dfs}
    expected_ids: Set[str] = {
        f"K200_v{corpus_version}_seed42_{sample_start}_{sample_end}",
        f"K200_v{corpus_version}_seed43_{sample_start}_{sample_end}",
    }
    assert run_ids == expected_ids

    for df in dfs:
        assert (df["corpus_version"] == corpus_version).all()


def test_generate_topic_metadata_dfs_uses_explicit_seed_numbers() -> None:
    """
    Verify that `generate_topic_metadata_dfs` uses the provided seed_numbers
    when constructing `run_id` values for each run.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if each run's metadata DataFrame has a `run_id` that
        incorporates the corresponding seed number, corpus version, and date
        window.

    Raises
    ------
    AssertionError
        If any run's `run_id` does not match the expected pattern or if the
        returned list length does not equal the number of runs.

    Notes
    -----
    - Uses two runs with distinct seeds and simple top-words and coherence
      structures to make expectations explicit.
    """
    sample_start: dt.date = dt.date(2016, 8, 1)
    sample_end: dt.date = dt.date(2016, 8, 2)
    corpus_version: int = 3

    top_words_list: List[List[List[str]]] = [
        [["a", "b"], ["c", "d"]],
        [["e"], ["f"]],
    ]
    coherence_scores: List[List[float]] = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    seed_numbers: List[int] = [42, 43]

    dfs: List[pd.DataFrame] = cmu.generate_topic_metadata_dfs(
        sample_start_date=sample_start,
        sample_end_date=sample_end,
        top_words_list=top_words_list,
        coherence_scores=coherence_scores,
        seed_numbers=seed_numbers,
        corpus_version=corpus_version,
    )

    assert len(dfs) == 2

    expected_run_ids: Set[str] = {
        f"K200_v{corpus_version}_seed42_{sample_start}_{sample_end}",
        f"K200_v{corpus_version}_seed43_{sample_start}_{sample_end}",
    }
    run_ids: Set[str] = {df["run_id"].iloc[0] for df in dfs}
    assert run_ids == expected_run_ids

    # Check that topic_id is contiguous per run and lengths match input.
    lengths: List[int] = [len(df) for df in dfs]
    assert lengths == [2, 2]


def test_generate_topic_metadata_dfs_falls_back_to_default_seed_numbers() -> None:
    """
    Ensure that `generate_topic_metadata_dfs` falls back to DEFAULT_SEED_NUMBERS
    when seed_numbers is None and correctly zips seeds with runs.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the `run_id` values for each run incorporate the
        corresponding default seed numbers and date window.

    Raises
    ------
    AssertionError
        If the default seeds are not used, are misaligned with runs, or if
        the number of run_ids does not match the number of runs.

    Notes
    -----
    - Uses only the first two runs so that expectations only involve
      `DEFAULT_SEED_NUMBERS[0]` and `DEFAULT_SEED_NUMBERS[1]`.
    """
    sample_start: dt.date = dt.date(2017, 1, 1)
    sample_end: dt.date = dt.date(2017, 1, 2)
    corpus_version: int = 2

    top_words_list: List[List[List[str]]] = [
        [["g", "h"]],
        [["i", "j"]],
    ]
    coherence_scores: List[List[float]] = [
        [0.5],
        [0.6],
    ]

    default_seeds: List[int] = cmu.DEFAULT_SEED_NUMBERS
    assert len(default_seeds) >= 2

    dfs: List[pd.DataFrame] = cmu.generate_topic_metadata_dfs(
        sample_start_date=sample_start,
        sample_end_date=sample_end,
        top_words_list=top_words_list,
        coherence_scores=coherence_scores,
        seed_numbers=None,
        corpus_version=corpus_version,
    )

    assert len(dfs) == 2

    run_ids: Set[str] = {df["run_id"].iloc[0] for df in dfs}
    expected_run_ids: Set[str] = {
        f"K200_v{corpus_version}_seed{default_seeds[0]}_{sample_start}_{sample_end}",
        f"K200_v{corpus_version}_seed{default_seeds[1]}_{sample_start}_{sample_end}",
    }
    assert run_ids == expected_run_ids
