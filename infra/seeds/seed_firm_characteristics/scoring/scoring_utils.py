"""
Purpose
-------
Shared scoring policies and helpers for mapping (ticker → window → CIK) evidence
into comparable scores, plus split-decision logic. Kept independent of any class
state so these functions can be unit-tested and reused from multiple contexts.

Key behaviors
-------------
- Map SEC form types to (half-life, base-trust) weights.
- Apply exponential time-decay using a day-level policy clock.
- Compute per-row scores (single or vectorized) with an anchor-quality multiplier.
- Provide confidence label mapping for posteriors.
- Evaluate whether a mid-window split is justified and compute a split timestamp.

Conventions
-----------
- Timestamps are timezone-aware; callers should normalize inputs to UTC.
- Time-decay uses floor-to-day only inside score computation; stored evidence
  timestamps keep full resolution.
- Validity windows are half-open [start, end).
- Form-type universe is closed and normalized upstream.

Downstream usage
----------------
Import from seeding/mapping routines and from ConfidenceScores to compute scores,
decide on mid-window splits, and convert probabilities into coarse confidence labels.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from infra.seeds.seed_firm_characteristics.records.raw_record import RawRecord
from infra.seeds.seed_firm_characteristics.records.table_records import MappingEvidence

SPLIT_COUNT_THRESHOLD: int = 3
SPLIT_COUNT_FRACTION: float = 0.3
SPLIT_DAY_THRESHOLD: int = 10


@dataclass(slots=True)
class CandidateData:
    """
    Purpose
    -------
    Hold per-(ticker, candidate_cik) aggregate state used to compute posteriors
    and to reference a representative evidence row.

    Key behaviors
    -------------
    - Accumulates the running `score` and a `count` of deduped evidence rows.
    - Tracks `max_score` (largest single-evidence contribution) and its
      `evidence_id` for provenance.

    Parameters
    ----------
    score : float
        Running sum of contributions from all deduped evidence rows.
    count : int
        Number of deduped evidence rows merged into `score`.
    max_score : float
        Largest single-evidence contribution observed so far.
    evidence_id : str
        The `evidence_id` associated with `max_score`.

    Notes
    -----
    - Mutable by design; used as a lightweight accumulator.
    - `slots=True` reduces per-instance memory overhead.
    """

    score: float
    count: int
    first_evidence_date: pd.Timestamp
    last_evidence_date: pd.Timestamp
    max_score: float
    evidence_id: str


@dataclass
class BestCandidateResult:
    """
    Purpose
    -------
    Lightweight container for the winning candidate within a (ticker, window).

    Fields
    ------
    candidate_cik : str
        Selected CIK.
    posterior_mean : float
        Posterior probability for `candidate_cik` in [0, 1].
    candidate_data : CandidateData
        Aggregates (score, count, max_score, evidence_id, envelope dates).

    Notes
    -----
    - Used by loaders and split logic to reference the representative evidence.
    """

    candidate_cik: str
    posterior_mean: float
    candidate_data: CandidateData


@dataclass
class MidWindowSplitData:
    """
    Purpose
    -------
    Result of a mid-window split decision between the best and second-best candidates.

    Fields
    ------
    split_date : pandas.Timestamp
        UTC day boundary at which to split the window.
    best_first : bool
        True if the best candidate’s envelope precedes the second-best; else False.
    second_best_candidate_result : BestCandidateResult
        Runner-up candidate details for constructing the second episode.

    Notes
    -----
    - Callers use `best_first` to order the two resulting [start, split) / [split, end) episodes.
    """

    split_date: pd.Timestamp
    best_first: bool
    second_best_candidate_result: BestCandidateResult


def calculate_score(evidence: MappingEvidence) -> float:
    """
    Compute the per-evidence contribution given recency decay, form trust, and
    anchor quality.

    Parameters
    ----------
    evidence : MappingEvidence
        Evidence with tz-aware `filed_at`, a non-null half-open validity window
        [start, end), `form_type`, and `raw_record`.

    Returns
    -------
    float
        Score = base_trust * exp(-ln(2) * Δt / half_life) * quality,
        where Δt = max(window_end - floor_UTC(filed_at_day), 0) in days.

    Raises
    ------
    None

    Notes
    -----
    - Scoring composition and timing:
        - Combine three interpretable factors:
            score = base_trust(form) * exp(-ln(2) * Δt / half_life(form)) * quality(anchor)
        - Δt is measured from floor_UTC(filed_at) to the validity window end; this matches
            the as-of policy (“evidence remains informative until window_end”).
        - Form hierarchy: listings/registrations decay slowest (most durable identity),
            annuals next, then registrations for offerings/M&A/transition, then quarterlies,
            then event-driven current reports.
        - Anchor quality gives a small, targeted haircut when you had to use a fallback
            text file instead of the canonical complete submission.
    """

    floor_filed_at: pd.Timestamp = evidence.filed_at.floor("D")
    time_delta: int = max((evidence.validity_window[-1] - floor_filed_at).days, 0)
    half_life, base_trust = calculate_form_type_effect(evidence.form_type)
    quality_score = calculate_quality_score(evidence.raw_record)
    decay_rate: float = np.log(2) / half_life
    return base_trust * np.exp(-decay_rate * time_delta) * quality_score


def calculate_form_type_effect(form_type: str) -> tuple[float, float]:
    """
    Map a filing form type to (half_life_days, base_trust_weight).

    Parameters
    ----------
    form_type : str
        Normalized SEC form type (e.g., "10-K", "8-A12B", "6-K/A").

    Returns
    -------
    tuple[int, float]
        (half_life_days, base_trust), e.g. (720, 1.0) for 10-K.

    Raises
    ------
    None

    Notes
    -----
    - Rationale for form-type weights and half-lives:
        - 8-A12B / 8-A12G (+ amendments): Longest half-life (36m), base_trust=1.0.
            These register a class of securities for exchange trading.
            They pin down the issuer’s legal identity at the listing level and change
            rarely; hence, they remain informative about identity for the longest.
        - 10-K / 20-F / 40-F (+ amendments): Long half-life (24m), base_trust=1.0.
            Annual reports are comprehensive, audited (10-K) or equivalent (20-F/40-F),
            and contain definitive issuer metadata (CIK, name, fiscal info). They
            refresh annually, so slightly shorter than 8-A registrations.
        - S-1 / S-3 / S-4 (+ amendments) / 10-KT: Medium half-life (12m), base_trust=0.8.
            Registration statements (and transition 10-KT) are highly informative
            around IPOs, follow-on offerings, and reorgs/M&A, but are episodic, often
            amended, and can include forward-looking or interim identity states.
        - 10-Q / 10-QT (+ amendments): Medium-short half-life (9m), base_trust=0.9.
            Quarterly reports are reviewed (not fully audited), contain less identity
            detail than annuals, and are superseded quickly by subsequent quarters.
        - 8-K / 6-K (+ amendments): Shortest half-life (6m), base_trust=0.6.
            Event-driven (“current”) reports are heterogeneous. Some items (e.g., name/
            ticker changes) are critical, but many are noisy. We keep
            a short half-life and lower base trust to reflect higher variance in signal.
    """

    match form_type:
        case "8-A12B" | "8-A12B/A" | "8-A12G" | "8-A12G/A":
            half_life = 1080  # 36 months
            base_trust = 1.0
        case "10-K" | "10-K/A" | "20-F" | "20-F/A" | "40-F" | "40-F/A":
            half_life = 720  # 24 months
            base_trust = 1.0
        case "S-1" | "S-1/A" | "S-3" | "S-3/A" | "S-4" | "S-4/A" | "10-KT":
            half_life = 360  # 12 months
            base_trust = 0.8
        case "10-Q" | "10-QT" | "10-Q/A":
            half_life = 270  # 9 months
            base_trust = 0.9
        case "8-K" | "8-K/A" | "6-K" | "6-K/A":
            half_life = 180  # 6 months
            base_trust = 0.6
    return half_life, base_trust


def calculate_quality_score(raw_record: RawRecord) -> float:
    """
    Return a quality multiplier based on the filing-page anchor used.

    Parameters
    ----------
    raw_record : RawRecord
        Raw record containing `filing_page.anchor_used` ∈
        {"complete-submission", "any-txt-file"}.

    Returns
    -------
    float
        1.0 for "complete-submission"; 0.9 for "any-txt-file".

    Raises
    ------
    None
        Assumes `anchor_used` is always present and one of the two values.

    Notes
    -----
    - Anchor-based quality multiplier:
        - "complete-submission" → 1.0: Canonical “Complete submission text file”. Includes
            full EDGAR header block; lowest risk of missing header fields.
        - "any-txt-file" → 0.9: Fallback when the canonical link is unavailable. Can be an
            exhibit or partial text; header fields may be incomplete or formatted irregularly.
            Apply a small penalty to reflect higher extraction risk.
    """

    filings_page = raw_record.get("filing_page")
    if filings_page is None:
        return 0.0  # Only for typing safety; should not happen in practice.
    anchor_used = filings_page.get("anchor_used")
    if anchor_used is None:
        return 0.0  # Only for typing safety; should not happen in practice.
    quality_score: float
    match anchor_used:
        case "complete-submission":
            quality_score = 1.0
        case "any-txt-file":
            quality_score = 0.9
    return quality_score


def calculate_batch_score(evidence_df: pd.DataFrame) -> pd.Series:
    """
    Vectorized per-row score computation for a dataframe of evidences.

    Parameters
    ----------
    evidence_df : pandas.DataFrame
        Columns required: ['validity_window', 'filed_at', 'form_type', 'raw_record'].
        `validity_window[i][1]` (end) must be non-null; `filed_at` tz-aware.

    Returns
    -------
    pandas.Series
        Score per row, index-aligned with `evidence_df`.

    Raises
    ------
    None

    Notes
    -----
    - Uses vectorized floor-to-day on `filed_at`, maps form types to
    (half_life, base_trust), derives a quality multiplier from anchors,
    then applies exponential decay.
    """

    window_end: pd.Series = evidence_df["validity_window"].map(lambda x: x[1])
    filed_floor: pd.Series = evidence_df["filed_at"].dt.floor("D")
    time_delta: pd.Series = ((window_end - filed_floor).dt.days).clip(lower=0)
    evidence_df["half_life"], evidence_df["base_trust_col"] = batch_form_type_effect(
        evidence_df["form_type"]
    )
    evidence_df["quality_score"] = calculate_batch_quality_score(evidence_df["raw_record"])
    evidence_df["decay_rate"] = np.log(2) / evidence_df["half_life"]
    score: pd.Series = (
        evidence_df["base_trust_col"]
        * np.exp(-evidence_df["decay_rate"] * time_delta)
        * evidence_df["quality_score"]
    )
    return score


def calculate_batch_quality_score(raw_records: pd.Series) -> pd.Series:
    """
    Vectorized quality multiplier extraction from raw records.

    Parameters
    ----------
    raw_records : pandas.Series
        Series of RawRecord dicts containing `filing_page.anchor_used`.

    Returns
    -------
    pandas.Series
        Quality multipliers (1.0 or 0.9), aligned to `raw_records.index`.

    Raises
    ------
    None

    Notes
    -----
    - Vectorized application of the anchor-quality policy:
        - "complete-submission" gets 1.0, "any-txt-file" gets 0.9.
        - Assumes `filing_page.anchor_used` is always present; the scoring penalty only
            activates for the explicit fallback case.

    """

    quality_score_mask = {"complete-submission": 1.0, "any-txt-file": 0.9}
    return (
        raw_records.map(lambda x: x.get("filing_page").get("anchor_used"))
        .map(quality_score_mask)
        .fillna(1.0)
    )


def batch_form_type_effect(form_types: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Vectorized mapping from form types to (half_life_days, base_trust_weight).

    Parameters
    ----------
    form_types : pandas.Series
        Series of normalized SEC form types.

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        (half_life_days, base_trust) as Series aligned to `form_types.index`.

    Raises
    ------
    None

    Notes
    -----
    - Same scoring policy as `calculate_form_type_effect`, vectorized:
        - 8-A12B / 8-A12G (+ amendments): 36m half-life, base_trust=1.0 — listing/class
            registration is the most stable identity signal.
        - 10-K / 20-F / 40-F (+ amendments): 24m half-life, base_trust=1.0 — audited,
            comprehensive, refreshed annually.
        - S-1 / S-3 / S-4 (+ amendments) / 10-KT: 12m half-life, base_trust=0.8 — episodic,
            often amended; strong but less durable than annuals.
        - 10-Q / 10-QT (+ amendments): 9m half-life, base_trust=0.9 — reviewed, fast-obsoleting.
        - 8-K / 6-K (+ amendments): 6m half-life, base_trust=0.6 — heterogeneous, higher noise.
    """

    half_life_mask = {
        "8-A12B": 1080,
        "8-A12B/A": 1080,
        "8-A12G": 1080,
        "8-A12G/A": 1080,
        "10-K": 720,
        "10-K/A": 720,
        "20-F": 720,
        "20-F/A": 720,
        "40-F": 720,
        "40-F/A": 720,
        "S-1": 360,
        "S-1/A": 360,
        "S-3": 360,
        "S-3/A": 360,
        "S-4": 360,
        "S-4/A": 360,
        "10-KT": 360,
        "10-Q": 270,
        "10-QT": 270,
        "10-Q/A": 270,
        "8-K": 180,
        "8-K/A": 180,
        "6-K": 180,
        "6-K/A": 180,
    }
    base_trust_mask = {
        "8-A12B": 1.0,
        "8-A12B/A": 1.0,
        "8-A12G": 1.0,
        "8-A12G/A": 1.0,
        "10-K": 1.0,
        "10-K/A": 1.0,
        "20-F": 1.0,
        "20-F/A": 1.0,
        "40-F": 1.0,
        "40-F/A": 1.0,
        "S-1": 0.8,
        "S-1/A": 0.8,
        "S-3": 0.8,
        "S-3/A": 0.8,
        "S-4": 0.8,
        "S-4/A": 0.8,
        "10-KT": 0.8,
        "10-Q": 0.9,
        "10-QT": 0.9,
        "10-Q/A": 0.9,
        "8-K": 0.6,
        "8-K/A": 0.6,
        "6-K": 0.6,
        "6-K/A": 0.6,
    }
    return form_types.map(half_life_mask), form_types.map(base_trust_mask)


def confidence_to_label(confidence: float) -> str:
    """
    Map a posterior probability to a qualitative confidence label.

    Parameters
    ----------
    confidence : float
        Posterior probability in [0, 1].

    Returns
    -------
    str
        One of {"high", "medium", "low"} based on fixed thresholds.

    Notes
    -----
    - Thresholds: high ≥ 0.8; medium ≥ 0.3; else low.
    - Keep thresholds in configuration if you expect to revisit them.
    """
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.3:
        return "medium"
    else:
        return "low"


def check_split_window_conditions(
    best_candidate_data: CandidateData,
    second_best_candidate_data: CandidateData,
) -> bool:
    """
    Decide whether evidence supports splitting a window into two CIK episodes.

    Parameters
    ----------
    best_candidate_data : CandidateData
        Aggregate for the top candidate in the window.
    second_best_candidate_data : CandidateData
        Aggregate for the runner-up candidate.

    Returns
    -------
    bool
        True if conditions justify a split:
        - Non-overlapping evidence envelopes between candidates;
        - Runner-up support ≥ max(SPLIT_COUNT_THRESHOLD, SPLIT_COUNT_FRACTION * best.count);
        - Runner-up evidence span in days ≥ SPLIT_DAY_THRESHOLD.

    Raises
    ------
    None

    Notes
    -----
    - Overlap check uses first/last evidence timestamps, not decay-adjusted scores.
    - Thresholds are defined at module scope for auditability.
    """

    if best_candidate_data is None or second_best_candidate_data is None:
        return False  # Only for typing
    if second_best_candidate_data.first_evidence_date < best_candidate_data.last_evidence_date:
        if second_best_candidate_data.last_evidence_date > best_candidate_data.first_evidence_date:
            return False
    if second_best_candidate_data.count < max(
        SPLIT_COUNT_THRESHOLD, SPLIT_COUNT_FRACTION * best_candidate_data.count
    ):
        return False
    second_best_time_span = (
        second_best_candidate_data.last_evidence_date
        - second_best_candidate_data.first_evidence_date
    ).days
    if second_best_time_span < SPLIT_DAY_THRESHOLD:
        return False
    return True


def calculate_split_date(
    best_candidate_data: CandidateData, second_best_candidate_data: CandidateData
) -> tuple[pd.Timestamp, bool]:
    """
    Compute a mid-gap split timestamp and which candidate appears first.

    Parameters
    ----------
    best_candidate_data : CandidateData
        Aggregate for the current argmax candidate.
    second_best_candidate_data : CandidateData
        Aggregate for the runner-up candidate.

    Returns
    -------
    tuple[pandas.Timestamp, bool]
        (split_date_UTC_floored_to_day, best_cik_first_flag).

    Raises
    ------
    None

    Notes
    -----
    - Picks the chronological order from the disjoint evidence envelopes and sets
      the split as the midpoint between the close edges, floored to day.
    - If midpoint floors to the earlier edge, adds 1 day to preserve a non-empty
      right-hand side when the caller builds [start, split) and [split, end).
    """

    first_candidate: CandidateData
    last_candidate: CandidateData
    best_first: bool
    if best_candidate_data.last_evidence_date < second_best_candidate_data.first_evidence_date:
        first_candidate = best_candidate_data
        last_candidate = second_best_candidate_data
        best_first = True
    else:
        first_candidate = second_best_candidate_data
        last_candidate = best_candidate_data
        best_first = False
    delta: pd.Timedelta = (
        last_candidate.first_evidence_date - first_candidate.last_evidence_date
    ) / 2
    split_date: pd.Timestamp = (first_candidate.last_evidence_date + delta).floor("D")
    if split_date == first_candidate.last_evidence_date:
        split_date += pd.Timedelta(days=1)
    return (split_date, best_first)
