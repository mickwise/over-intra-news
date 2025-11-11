"""
Purpose
-------
Tests for the adjudication_multi_candidate_utils module.

Key behaviors
-------------
- Verify periodic-form feature engineering and auto-accept rule masks for
  single-candidate and multi-candidate scenarios.
- Verify canonical evidence selection obeys the configured form priority
  and date ordering, with and without auto-accept masks.
- Verify manual candidate overrides correctly remove specified
  (ticker, candidate_cik) rows from the evidence DataFrame.

Conventions
-----------
- Tests construct small, fully in-memory pandas DataFrames to exercise
  the adjudication logic without any external dependencies.
- Configuration constants (e.g., PERIODIC_FORMS, PERIODIC_FORMS_EVIDENCE_HIERARCHY,
  MANUALLY_OVERRIDDEN_CIKS) may be patched in-place on the target module
  when necessary to keep expectations explicit and stable.

Downstream usage
----------------
Run this test module with pytest (locally and in CI) to guard refactors of
the auto-accept rules, canonical evidence selection, and manual override
behavior in the adjudication pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from notebooks_utils.data_notebooks_utils.adjudication_multi_candidate import (
    adjudication_multi_candidate_utils,
)


def test_calculate_form_type_features_respects_periodic_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that calculate_form_type_features uses PERIODIC_FORMS to set has_periodic.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch PERIODIC_FORMS on the target module so
        the expected periodic behavior is fully deterministic within the test.

    Returns
    -------
    None
        The test passes if has_periodic is True only for rows whose form_type
        is present in the patched PERIODIC_FORMS set.

    Raises
    ------
    AssertionError
        If has_periodic is incorrectly set for any row.

    Notes
    -----
    - This test ensures that periodic detection logic is tied to the
      configuration constant rather than hard-coded form types.
    """
    monkeypatch.setattr(
        adjudication_multi_candidate_utils,
        "PERIODIC_FORMS",
        {"10-K"},
        raising=True,
    )
    df = pd.DataFrame(
        {
            "form_type": ["10-K", "8-K"],
            "ticker": ["AAA", "BBB"],
            "validity_window": ["w1", "w1"],
        }
    )

    result = adjudication_multi_candidate_utils.calculate_form_type_features(df)
    assert result.loc[0, "has_periodic"]
    assert not result.loc[1, "has_periodic"]


def test_extract_single_periodic_marks_only_single_candidate_groups() -> None:
    """
    Ensure extract_single_periodic flags only single-candidate groups with periodic evidence.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the mask is True only for rows belonging to
        (ticker, validity_window) groups that have exactly one candidate_cik
        and at least one periodic row.

    Raises
    ------
    AssertionError
        If groups with multiple candidates or without periodic evidence are
        incorrectly flagged as single_periodic_mask=True.

    Notes
    -----
    - The test constructs three groups:
      - One candidate with periodic evidence (should be True).
      - One candidate with no periodic evidence (should be False).
      - Two candidates, one periodic and one not (should be False).
    """

    df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "CCC"],
            "validity_window": ["w1", "w1", "w1", "w1"],
            "candidate_cik": ["C1", "C2", "C3", "C4"],
            "has_periodic": [True, False, True, False],
        }
    )
    mask = adjudication_multi_candidate_utils.extract_single_periodic(
        df, keys=["ticker", "validity_window"]
    )
    assert mask.tolist() == [True, False, False, False]


def test_multi_candidate_single_periodic_auto_accept_marks_only_unique_periodic_candidate() -> None:
    """
    Ensure multi_candidate_single_periodic_auto_accept flags only the unique periodic candidate.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the mask is True only for rows where:
        - the (ticker, validity_window) group has multiple candidates, and
        - exactly one candidate has any periodic evidence.

    Raises
    ------
    AssertionError
        If groups with zero or multiple periodic candidates are incorrectly
        flagged, or if rows for the non-periodic candidates in a winning group
        are marked True.

    Notes
    -----
    - The test constructs three multi-candidate groups:
      - Group with exactly one periodic candidate (should be True for that
        candidate only).
      - Group with two periodic candidates (should be False for all rows).
      - Group with no periodic candidates (should be False for all rows).
    """
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "validity_window": ["w1"] * 6,
            "candidate_cik": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "has_periodic": [True, False, True, True, False, False],
        }
    )
    mask = adjudication_multi_candidate_utils.multi_candidate_single_periodic_auto_accept(
        df, keys=["ticker", "validity_window"]
    )
    assert mask.tolist() == [True, False, False, False, False, False]


def test_find_canonical_evidence_prioritizes_forms_and_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that find_canonical_evidence respects form priority and filed_at ordering.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch PERIODIC_FORMS_EVIDENCE_HIERARCHY on the
        target module so that form-score ordering is explicit.

    Returns
    -------
    None
        The test passes if the selected canonical evidence_id corresponds to
        the row with:
        - the lowest form_score,
        - the latest filed_at among ties,
        - and the highest evidence_id as a final tie-breaker.

    Raises
    ------
    AssertionError
        If any lower-priority or older evidence is incorrectly chosen as
        canonical.

    Notes
    -----
    - The test constructs multiple evidence rows for a single candidate and
      expects the best 10-K with the latest date to be selected.
    """
    monkeypatch.setattr(
        adjudication_multi_candidate_utils,
        "PERIODIC_FORMS_EVIDENCE_HIERARCHY",
        {"10-K": 0, "10-Q": 1},
        raising=True,
    )
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "validity_window": ["w1", "w1", "w1"],
            "candidate_cik": ["C1", "C1", "C1"],
            "form_type": ["10-Q", "10-K", "10-K"],
            "filed_at": pd.to_datetime(["2021-01-01", "2021-01-15", "2021-02-01"]),
            "evidence_id": ["E1", "E2", "E3"],
        }
    )
    canonical = adjudication_multi_candidate_utils.find_canonical_evidence(
        df, keys=["ticker", "validity_window"], auto_accept_mask=None
    )
    assert set(canonical.values) == {"E3"}


def test_filter_auto_accept_applies_rules_and_sets_winners() -> None:
    """
    Verify that filter_auto_accept sets rule flags, winners, and canonical evidence.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
        - Rule 1 is applied to single-candidate periodic groups.
        - Rule 2 is applied to multi-candidate groups with a unique periodic candidate.
        - winning_candidate and canonical_evidence_id are set consistently
          for all auto-accepted rows.
        - source is set to 'edgar_fts' for auto-accepted rows.

    Raises
    ------
    AssertionError
        If any of the rule masks, winning_candidate values, canonical
        evidence_ids, or source fields are inconsistent with the expected
        behavior.

    Notes
    -----
    - The test constructs:
      - AAA/w1: single candidate with periodic evidence (Rule 1).
      - BBB/w1: two candidates, exactly one periodic (Rule 2).
      - CCC/w1: two non-periodic candidates (no auto-accept).
    """
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "validity_window": ["w1"] * 6,
            "candidate_cik": ["C1", "C1", "C2", "C3", "C4", "C5"],
            "form_type": ["10-K", "8-K", "10-K", "8-K", "8-K", "8-K"],
            "filed_at": pd.to_datetime(
                [
                    "2021-01-10",
                    "2021-01-05",
                    "2021-02-01",
                    "2021-01-20",
                    "2021-03-01",
                    "2021-03-02",
                ]
            ),
            "evidence_id": ["E1", "E2", "E3", "E4", "E5", "E6"],
        }
    )
    result = adjudication_multi_candidate_utils.filter_auto_accept(df)

    # AAA/w1: Rule 1 (single candidate with periodic evidence).
    aaa_mask = (result["ticker"] == "AAA") & (result["validity_window"] == "w1")
    aaa = result.loc[aaa_mask]

    assert aaa["rule_1"].all()
    assert not aaa["rule_2"].any()
    assert (aaa["winning_candidate"] == "C1").all()

    # Among AAA rows, exactly one canonical_evidence_id, and it should be E1.
    aaa_canon = aaa["canonical_evidence_id"].dropna().unique()
    assert len(aaa_canon) == 1
    assert aaa_canon[0] == "E1"
    assert aaa["canonical_evidence_id"].notna().sum() == 1
    assert (aaa["source"].dropna().unique() == ["edgar_fts"]).all()

    # BBB/w1: Rule 2 (unique periodic candidate among many).
    bbb_mask = (result["ticker"] == "BBB") & (result["validity_window"] == "w1")
    bbb = result.loc[bbb_mask]

    # Only the periodic candidate C2 should be winning.
    winners_bbb = bbb["winning_candidate"].dropna().unique()
    assert set(winners_bbb) == {"C2"}
    assert bbb["rule_2"].any()

    # canonical_evidence_id for BBB group should be E3, exactly once.
    bbb_canon = bbb["canonical_evidence_id"].dropna().unique()
    assert len(bbb_canon) == 1
    assert bbb_canon[0] == "E3"
    assert bbb["canonical_evidence_id"].notna().sum() == 1
    assert (bbb["source"].dropna().unique() == ["edgar_fts"]).all()

    # CCC/w1: no periodic evidence, no auto-accept.
    ccc_mask = (result["ticker"] == "CCC") & (result["validity_window"] == "w1")
    ccc = result.loc[ccc_mask]

    assert not ccc["rule_1"].any()
    assert not ccc["rule_2"].any()
    assert ccc["winning_candidate"].isna().all()
    assert ccc["canonical_evidence_id"].isna().all()
    assert ccc["source"].isna().all()


def test_remove_overridden_candidates_filters_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure remove_overridden_candidates drops rows matching MANUALLY_OVERRIDDEN_CIKS.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch MANUALLY_OVERRIDDEN_CIKS on the target
        module so that the override behavior is controlled and explicit.

    Returns
    -------
    None
        The test passes if all rows whose (ticker, candidate_cik) combinations
        appear in the patched MANUALLY_OVERRIDDEN_CIKS mapping are removed and
        the remaining rows are preserved.

    Raises
    ------
    AssertionError
        If any overridden rows remain in the output or if non-overridden rows
        are mistakenly removed.

    Notes
    -----
    - The test patches MANUALLY_OVERRIDDEN_CIKS with a single override for
      ticker 'AAA' and candidate_cik 'C1' and verifies that only those rows
      are filtered out.
    """
    monkeypatch.setattr(
        adjudication_multi_candidate_utils,
        "MANUALLY_OVERRIDDEN_CIKS",
        {"AAA": ["C1"]},
        raising=True,
    )
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "candidate_cik": ["C1", "C2", "C3"],
            "form_type": ["10-K", "10-K", "10-K"],
            "filed_at": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            "evidence_id": ["E1", "E2", "E3"],
        }
    )
    result = adjudication_multi_candidate_utils.remove_overridden_candidates(df)
    assert (result["ticker"].tolist(), result["candidate_cik"].tolist()) == (
        ["AAA", "BBB"],
        ["C2", "C3"],
    )
