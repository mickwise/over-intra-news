"""
Purpose
-------
Rules and helpers for auto-accepting ticker–CIK candidates based on SEC
filing evidence, with a focus on periodic forms (10-K, 10-Q, etc.) and
manual override rules.

Key behaviors
-------------
- Compute per-row features indicating whether evidence comes from periodic
  SEC forms.
- Identify (ticker, validity_window) pairs that can be auto-accepted based
  on simple periodic-form heuristics.
- Select a canonical evidence_id per winning candidate for downstream
  profile history loading.
- Remove manually overridden candidate CIKs based on a curated override map.

Conventions
-----------
- Evidence data is provided as a pandas DataFrame with, at minimum, the
  columns: 'ticker', 'validity_window', 'candidate_cik', 'form_type',
  'filed_at', and 'evidence_id'.
- PERIODIC_FORMS and PERIODIC_FORMS_EVIDENCE_HIERARCHY define which SEC
  forms are considered periodic and how they are prioritized when choosing
  canonical evidence.
- MANUALLY_OVERRIDDEN_CIKS contains ticker → [candidate_cik] mappings that
  should be removed before auto-accept logic is applied.

Downstream usage
----------------
Use these helpers in the adjudication pipeline to:
- Filter and annotate evidence frames with auto-accept decisions.
- Select canonical evidence records for winning candidates.
- Apply manual overrides before final ticker–CIK decisions are materialized.
"""

from typing import List

import pandas as pd

# fmt: off
from notebooks_utils.data_notebooks_utils.adjudication_multi_candidate.\
    adjudication_multi_candidate_config import (
    MANUALLY_OVERRIDDEN_CIKS,
    PERIODIC_FORMS,
    PERIODIC_FORMS_EVIDENCE_HIERARCHY,
)

# fmt: on


def filter_auto_accept(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply auto-accept rules to an evidence DataFrame and annotate winners.

    Parameters
    ----------
    evidence_df : pandas.DataFrame
        Evidence-level DataFrame containing at least:
        - 'ticker'
        - 'validity_window'
        - 'candidate_cik'
        - 'form_type'
        - 'filed_at'
        - 'evidence_id'

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with additional columns:
        - 'has_periodic' : bool
            Whether any periodic form supports the row's (ticker, validity_window, candidate_cik).
        - 'winning_candidate' : object
            The candidate_cik chosen as winner for the (ticker, validity_window) pair,
            or NA if not auto-accepted.
        - 'canonical_evidence_id' : object
            The evidence_id selected as canonical for the winning candidate, or NA.
        - 'rule_1' : bool
            True if the row is accepted via the single-candidate periodic rule.
        - 'rule_2' : bool
            True if the row is accepted via the multi-candidate single-periodic rule.
        - 'source' : object
            Populated with 'edgar_fts' for auto-accepted rows; NA otherwise.

    Raises
    ------
    KeyError
        If required columns (e.g., 'ticker', 'validity_window', 'candidate_cik',
        'form_type', 'filed_at', 'evidence_id') are missing from evidence_df.

    Notes
    -----
    - This function orchestrates feature computation, rule masks, and canonical
      evidence selection; it does not modify the original DataFrame in place.
    - Auto-accept rules currently consist of:
        - Rule 1: Exactly one candidate per (ticker, validity_window), with at
            least one periodic form.
        - Rule 2: Multiple candidates per pair, but exactly one candidate has
            any periodic evidence.
    """

    keys: List[str] = ["ticker", "validity_window"]
    evidence_df = calculate_form_type_features(evidence_df)
    evidence_df["winning_candidate"] = pd.NA
    evidence_df["canonical_evidence_id"] = pd.NA
    evidence_df["rule_1"] = False
    evidence_df["rule_2"] = False
    evidence_df["source"] = pd.NA
    single_periodic_mask: pd.Series = extract_single_periodic(evidence_df, keys)
    multi_candidate_single_periodic_mask: pd.Series = multi_candidate_single_periodic_auto_accept(
        evidence_df, keys
    )
    evidence_df.loc[single_periodic_mask, "rule_1"] = True
    evidence_df.loc[multi_candidate_single_periodic_mask, "rule_2"] = True
    auto_accept_mask: pd.Series = single_periodic_mask | multi_candidate_single_periodic_mask
    evidence_df.loc[auto_accept_mask, "winning_candidate"] = evidence_df.loc[
        auto_accept_mask, "candidate_cik"
    ]
    canonical_evidence_id_series: pd.Series = find_canonical_evidence(
        evidence_df, keys, auto_accept_mask
    )
    evidence_df.loc[auto_accept_mask, "canonical_evidence_id"] = canonical_evidence_id_series.loc[
        auto_accept_mask
    ]
    evidence_df.loc[auto_accept_mask, "source"] = "edgar_fts"
    return evidence_df


def calculate_form_type_features(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate evidence rows with a 'has_periodic' feature based on form_type.

    Parameters
    ----------
    evidence_df : pandas.DataFrame
        Evidence-level DataFrame containing a 'form_type' column used to detect
        whether a row is backed by a periodic SEC form.

    Returns
    -------
    pandas.DataFrame
        A DataFrame equivalent to the input but with an additional boolean
        column:
        - 'has_periodic' : True if 'form_type' is in PERIODIC_FORMS; False otherwise.

    Raises
    ------
    KeyError
        If the 'form_type' column is missing from evidence_df.

    Notes
    -----
    - The set of periodic forms is defined by PERIODIC_FORMS in the configuration
      module and can be adjusted centrally as needed.
    - This helper is intended to be called early in the pipeline before any
      auto-accept rules that depend on periodic evidence.
    """

    evidence_df = evidence_df.assign(
        has_periodic=evidence_df["form_type"].isin(PERIODIC_FORMS),
    )
    return evidence_df


def extract_single_periodic(evidence_df: pd.DataFrame, keys: List[str]) -> pd.Series:
    """
    Identify (ticker, validity_window) pairs with a single candidate and periodic evidence.

    Parameters
    ----------
    evidence_df : pandas.DataFrame
        Evidence-level DataFrame with at least:
        - key columns defined by `keys` (e.g., 'ticker', 'validity_window')
        - 'candidate_cik'
        - 'has_periodic' (boolean flag indicating periodic evidence)
    keys : list[str]
        Column names that define the group key for a potential mapping episode,
        typically ['ticker', 'validity_window'].

    Returns
    -------
    pandas.Series
        Boolean Series indexed like evidence_df, named 'single_periodic_mask',
        where True marks rows belonging to (ticker, validity_window) groups that:
        - have exactly one distinct candidate_cik, and
        - have at least one row with has_periodic = True.

    Raises
    ------
    KeyError
        If any of the key columns, 'candidate_cik', or 'has_periodic' are
        missing from evidence_df.

    Notes
    -----
    - This mask is used as "Rule 1" in the auto-accept pipeline: if there is
      only one candidate and it has periodic evidence, that candidate can be
      auto-accepted for the pair.
    - The function operates purely on group-level statistics and does not
      modify the input DataFrame.
    """

    single_pair_periodic: pd.DataFrame = evidence_df.groupby(keys, as_index=False).agg(
        number_of_candidates=("candidate_cik", "nunique"),
        has_periodic=("has_periodic", "any"),
    )
    accepted_pairs: pd.DataFrame = single_pair_periodic.loc[
        (single_pair_periodic["number_of_candidates"] == 1)
        & (single_pair_periodic["has_periodic"]),
        keys,
    ].drop_duplicates()
    membership: pd.DataFrame = evidence_df[keys].merge(
        accepted_pairs, on=keys, how="left", indicator=True
    )
    return pd.Series(
        membership["_merge"].eq("both").to_numpy(),
        index=evidence_df.index,
        name="single_periodic_mask",
    )


def multi_candidate_single_periodic_auto_accept(
    evidence_df: pd.DataFrame, keys: List[str]
) -> pd.Series:
    """
    Identify rows where a single candidate has periodic evidence among many.

    Parameters
    ----------
    evidence_df : pandas.DataFrame
        Evidence-level DataFrame containing at least:
        - key columns defined by `keys` (e.g., 'ticker', 'validity_window')
        - 'candidate_cik'
        - 'has_periodic' (boolean flag indicating periodic evidence)
    keys : list[str]
        Column names that define the group key for a potential mapping episode,
        typically ['ticker', 'validity_window'].

    Returns
    -------
    pandas.Series
        Boolean Series indexed like evidence_df, named
        'multi_candidate_single_periodic_mask', where True marks rows belonging
        to (ticker, validity_window, candidate_cik) triples that:
        - occur in groups with more than one distinct candidate_cik, and
        - have exactly one candidate_cik whose evidence has has_periodic = True.

    Raises
    ------
    KeyError
        If any of the key columns, 'candidate_cik', or 'has_periodic' are
        missing from evidence_df.

    Notes
    -----
    - This mask represents "Rule 2" in the auto-accept pipeline: when multiple
      candidates exist but only one has periodic evidence, that candidate is
      favored.
    - The function works at the candidate level, then projects back to the
      original evidence_df index via a merge.
    """

    candidate_level = evidence_df.groupby(keys + ["candidate_cik"], as_index=False).agg(
        candidate_has_periodic=("has_periodic", "any")
    )

    candidate_counts = candidate_level.groupby(keys)["candidate_cik"].transform("nunique")
    candidate_level = candidate_level[candidate_counts > 1]

    # 3. Count how many candidates with periodic there are per group
    periodic_counts = candidate_level.groupby(keys)["candidate_has_periodic"].transform("sum")

    # 4. Winners: groups where exactly one candidate has periodic, take that candidate
    winners = candidate_level[candidate_level["candidate_has_periodic"] & (periodic_counts == 1)][
        keys + ["candidate_cik"]
    ]

    # 5. Mark all evidence rows belonging to those winning candidates
    membership = evidence_df[keys + ["candidate_cik"]].merge(
        winners, on=keys + ["candidate_cik"], how="left", indicator=True
    )

    return pd.Series(
        membership["_merge"].eq("both").to_numpy(),
        index=evidence_df.index,
        name="multi_candidate_single_periodic_mask",
    )


def find_canonical_evidence(
    evidence_df: pd.DataFrame, keys: List[str], auto_accept_mask: pd.Series | None = None
) -> pd.Series:
    """
    Select a canonical evidence_id per winning candidate based on form priority.

    Parameters
    ----------
    evidence_df : pandas.DataFrame
        Evidence-level DataFrame containing, at minimum:
        - key columns defined by `keys` (e.g., 'ticker', 'validity_window')
        - 'candidate_cik'
        - 'form_type'
        - 'filed_at'
        - 'evidence_id'
        - optionally 'winning_candidate' when auto_accept_mask is provided.
    keys : list[str]
        Column names that define the mapping episode, typically
        ['ticker', 'validity_window'].
    auto_accept_mask : pandas.Series or None, optional
        Boolean Series indexed like evidence_df indicating which rows belong
        to auto-accepted candidates. When provided, canonical evidence is
        chosen only within rows where:
        - auto_accept_mask is True, and
        - candidate_cik == winning_candidate.

    Returns
    -------
    pandas.Series
        Series named 'canonical_evidence_id', indexed like evidence_df,
        containing the selected evidence_id for each winning
        (ticker, validity_window, candidate_cik) triple. Non-winning rows
        are left as NaN and dropped before de-duplication.

    Raises
    ------
    KeyError
        If required columns (keys, 'candidate_cik', 'form_type', 'filed_at',
        'evidence_id') are missing from evidence_df.
    KeyError
        If auto_accept_mask is provided but 'winning_candidate' is absent
        from evidence_df.

    Notes
    -----
    - Form priority is determined by PERIODIC_FORMS_EVIDENCE_HIERARCHY, which
      maps form_type to an integer score. Lower scores are treated as higher
      priority.
    - Within each (keys + candidate_cik) group, canonical evidence is chosen
      by sorting by:
        - form_score ascending (best forms first),
        - filed_at descending (most recent first),
        - evidence_id descending (tie-breaker).
    - The returned Series includes only the canonical entries; non-winning
      indices are omitted via dropna().drop_duplicates().
    """

    group_keys: List[str] = keys + ["candidate_cik"]
    auto_accept_df: pd.DataFrame = evidence_df.copy()
    if auto_accept_mask is not None:
        winning_candidate_mask: pd.Series = auto_accept_mask & (
            evidence_df["candidate_cik"] == evidence_df["winning_candidate"]
        )
        auto_accept_df = auto_accept_df[winning_candidate_mask]
    auto_accept_df["form_score"] = auto_accept_df["form_type"].map(
        PERIODIC_FORMS_EVIDENCE_HIERARCHY
    )
    auto_accept_df.sort_values(
        ["form_score", "filed_at", "evidence_id"], ascending=[True, False, False], inplace=True
    )
    canonical_evidence: pd.DataFrame = auto_accept_df.groupby(
        group_keys, sort=False, as_index=False
    ).head(1)
    membership: pd.DataFrame = evidence_df[group_keys].merge(
        canonical_evidence[group_keys + ["evidence_id"]], on=group_keys, how="left", indicator=True
    )
    return (
        pd.Series(membership["evidence_id"], name="canonical_evidence_id", index=evidence_df.index)
        .dropna()
        .drop_duplicates()
    )


def remove_overridden_candidates(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop evidence rows for candidate CIKs that are manually overridden.

    Parameters
    ----------
    evidence_df : pandas.DataFrame
        Evidence-level DataFrame containing at least:
        - 'ticker'
        - 'candidate_cik'
        Rows matching any (ticker, candidate_cik) pairs in MANUALLY_OVERRIDDEN_CIKS
        will be removed.

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame where all rows corresponding to manually overridden
        candidate CIKs have been removed. The index is reset to a simple RangeIndex.

    Raises
    ------
    KeyError
        If 'ticker' or 'candidate_cik' columns are missing from evidence_df.

    Notes
    -----
    - MANUALLY_OVERRIDDEN_CIKS is a dict-like mapping from ticker to a list of
      candidate_cik values that should be excluded entirely from adjudication.
    - This function is typically applied early in the pipeline so that bad
      candidates do not participate in auto-accept logic or canonical evidence
      selection.
    """

    for ticker, candidates in MANUALLY_OVERRIDDEN_CIKS.items():
        override_mask: pd.Series = (evidence_df["ticker"] == ticker) & (
            evidence_df["candidate_cik"].isin(candidates)
        )
        evidence_df = evidence_df.loc[~override_mask].reset_index(drop=True)
    return evidence_df
