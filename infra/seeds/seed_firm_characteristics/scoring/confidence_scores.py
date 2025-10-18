"""
Purpose
-------
Aggregate time-decayed evidence for candidate (ticker → window → CIK) associations,
compute simple posteriors per (ticker, window), and select a single best CIK within
each window using a deterministic tie-break. Designed for small, clean batches where
most tickers/windows have a single plausible CIK.

Key behaviors
-------------
- Ingest single or batched `MappingEvidence` items.
- Convert a policy half-life (days) to an exponential decay rate and weight
  each evidence row by exp(-ln(2) * Δt / half_life), where
  Δt = max(window_end − floor_UTC(filed_at_day), 0) in days.
- Deduplicate batch inputs by `evidence_id` before accumulation.
- Maintain per-(ticker, window_key, candidate_cik) aggregates (total score, count,
  and the maximum single-evidence contribution with its `evidence_id`).
- Normalize to a posterior with a symmetric pseudo-count prior and return
  both posterior means and an argmax candidate per (ticker, window) with a deterministic tie-break.

Conventions
-----------
- Inputs are normalized upstream (tickers UPPER/trimmed, CIKs zero-padded).
- `filed_at` is timezone-aware; UTC conversions happen at the day boundary.
- The validity window is a half-open timestamp pair [start, end); only Δt to end
  is used here since filed_at ∈ [start, end) is already enforced.
- `half_life` is in days; `prior_pseudo_count` α is a small non-negative float.
- `window_key` is a canonical string derived from the half-open [start, end) window
  (e.g., "YYYY-MM-DD to YYYY-MM-DD").

Downstream usage
----------------
- Feed evidence via `evidence_to_score` (single) or `batch_evidence_to_scores`.
- Query `posterior_means(ticker, window_key)` or `best_candidate(ticker, window_key)`
  to drive writes into the curated mapping table; use the tracked representative
  `evidence_id` for provenance.
"""

from typing import cast

import pandas as pd

from infra.seeds.seed_firm_characteristics.records.table_records import (
    MappingEvidence,
    ValidityWindow,
    ValidityWindows,
    validity_window_to_str,
)
from infra.seeds.seed_firm_characteristics.scoring.scoring_utils import (
    BestCandidateResult,
    CandidateData,
    MidWindowSplitData,
    calculate_batch_score,
    calculate_score,
    calculate_split_date,
    check_split_window_conditions,
)

# ticker -> window_key -> candidate_cik -> CandidateData
ScoresAndCounts = dict[str, dict[str, dict[str, CandidateData]]]


class ConfidenceScores:
    """
    Purpose
    -------
    Maintain per-ticker, per-window aggregates of evidence-derived scores and compute
    a posterior over candidate CIKs using per-form half-lives (recency decay) and
    per-form/base quality weights. Selects a deterministic argmax within a window.

    Key behaviors
    -------------
    - Accumulates scores from single items or batched dataframes.
    - Applies per-form half-life (days) and a base-trust multiplier, then an
      anchor-quality multiplier from the raw record.
    - Computes posterior means with a symmetric pseudo-count and returns the
      best candidate with a deterministic tie-break (posterior, max_score, CIK).

    Attributes
    ----------
    scores_and_counts : dict[str, dict[str, dict[str, CandidateData]]]
        Nested mapping of ticker → window_key → candidate_cik → aggregates
        (score, count, max_score, evidence_id).
    prior_pseudo_count : float
        Pseudo-count α used for posterior smoothing.

    Notes
    -----
    - Form-specific half-lives and base trust are looked up at scoring time;
      there is no global half-life or stored decay_rate attribute.
    - Assumes upstream normalization (UPPER tickers, zero-padded CIKs) and
      that `filed_at` is tz-aware. `window_end` is required and non-null.
    """

    def __init__(
        self,
        ticker_validity_windows: dict[str, ValidityWindows],
        prior_pseudo_count: float = 0.1,
    ) -> None:
        self.scores_and_counts: ScoresAndCounts = {
            ticker: {validity_window_to_str(window): {} for window in windows}
            for ticker, windows in ticker_validity_windows.items()
        }
        self.prior_pseudo_count: float = prior_pseudo_count

    def evidence_to_score(self, evidence: MappingEvidence) -> None:
        """
        Accumulate a score for a single evidence row into the
        (ticker, window_key, candidate) bucket.

        Parameters
        ----------
        evidence : MappingEvidence
            Evidence containing ticker, candidate_cik, filed_at, validity_window,
            and evidence_id.

        Returns
        -------
        None

        Notes
        -----
        - Δt = max((window_end − floor_UTC(filed_at_day)), 0) in days.
        - Score = exp(-decay_rate * Δt) × base_trust(form) × quality(anchor).
        - The evidence’s `validity_window` is canonicalized to a `window_key` to
          select the per-window accumulator for this ticker.
        - Updates the per-(ticker, window_key, candidate) aggregates and representative
          evidence if this contribution exceeds the current `max_score`.
        """

        ticker: str = evidence.ticker
        validity_window: str = validity_window_to_str(evidence.validity_window)
        filed_at: pd.Timestamp = evidence.filed_at
        candidate_cik: str = evidence.candidate_cik
        score: float = calculate_score(evidence)
        self.update_scores_and_counts(
            ticker, validity_window, filed_at, candidate_cik, score, evidence.evidence_id
        )

    def batch_evidence_to_scores(self, evidences: list[MappingEvidence]) -> None:
        """
        Vectorized accumulation for a batch of evidence rows with deduplication.

        Parameters
        ----------
        evidences : list[MappingEvidence]
            Iterable of evidence records; duplicates by `evidence_id` are dropped
            (keeping the last) before aggregation.

        Returns
        -------
        None

        Notes
        -----
        - Computes scores via `calculate_score`, derives a canonical `window_key`
          from each row’s validity_window, groups by
          (ticker, window_key, filed_at, candidate_cik, evidence_id), and merges into state.
        """

        evidence_df: pd.DataFrame = pd.DataFrame([evidence.__dict__ for evidence in evidences])
        evidence_df["window_key"] = evidence_df["validity_window"].map(validity_window_to_str)
        evidence_df.drop_duplicates(subset=["evidence_id"], keep="last", inplace=True)
        evidence_df["score"] = calculate_batch_score(evidence_df)
        score_and_count_groups: pd.DataFrame = (
            evidence_df.groupby(
                ["ticker", "window_key", "filed_at", "candidate_cik", "evidence_id"]
            )
            .agg(score=("score", "sum"))
            .reset_index()
        )
        self.batch_update(score_and_count_groups)

    def batch_update(self, score_and_count_groups: pd.DataFrame) -> None:
        """
        Merge pre-aggregated (ticker, window_key, candidate_cik, evidence_id) scores into state.

        Parameters
        ----------
        score_and_count_groups : pandas.DataFrame
            Columns: ['ticker', 'window_key', 'candidate_cik', 'evidence_id', 'score'].

        Returns
        -------
        None

        Notes
        -----
        - Increments `count` once per distinct evidence_id.
        - Refreshes the representative evidence if `score` exceeds `max_score`.
        """

        for row in score_and_count_groups.itertuples(index=False):
            ticker: str = cast(str, row.ticker)
            validity_window: str = cast(str, row.window_key)
            filed_at: pd.Timestamp = cast(pd.Timestamp, row.filed_at)
            candidate_cik: str = cast(str, row.candidate_cik)
            evidence_id: str = cast(str, row.evidence_id)
            score: float = cast(float, row.score)
            self.update_scores_and_counts(
                ticker, validity_window, filed_at, candidate_cik, score, evidence_id
            )

    def update_scores_and_counts(
        self,
        ticker: str,
        validity_window: str,
        filed_at: pd.Timestamp,
        candidate_cik: str,
        score: float,
        evidence_id: str,
    ) -> None:
        """
        Merge a single evidence contribution into the
        per-(ticker, window_key, candidate) accumulator.

        Parameters
        ----------
        ticker : str
            Ticker key for the accumulator.
        validity_window : str
            Canonical window key, e.g. "YYYY-MM-DD to YYYY-MM-DD".
        filed_at : pd.Timestamp
            UTC-aware filing timestamp used to update first/last evidence dates.
        candidate_cik : str
            Candidate CIK receiving the contribution.
        score : float
            Evidence contribution after decay and weighting.
        evidence_id : str
            Deterministic ID of the evidence row (used for representative selection).

        Returns
        -------
        None

        Notes
        -----
        - Increments `count` once per call; deduplication should happen upstream.
        - Updates `first_evidence_date`/`last_evidence_date` to support split logic.
        - Replaces the representative evidence when `score` exceeds `max_score`.
        """

        if candidate_cik in self.scores_and_counts[ticker][validity_window]:
            current = self.scores_and_counts[ticker][validity_window].get(candidate_cik)
            if current is None:
                return None  # For typing safety; should not happen in practice.
            self.update_existing_scores_and_counts(current, score, evidence_id, filed_at)
        else:
            self.scores_and_counts[ticker][validity_window][candidate_cik] = CandidateData(
                score=score,
                count=1,
                first_evidence_date=filed_at,
                last_evidence_date=filed_at,
                max_score=score,
                evidence_id=evidence_id,
            )

    def update_existing_scores_and_counts(
        self,
        current_candidate_data: CandidateData,
        score: float,
        evidence_id: str,
        filed_at: pd.Timestamp,
    ) -> None:
        """
        Update an existing candidate’s aggregates with a new evidence contribution.

        Parameters
        ----------
        current_candidate_data : CandidateData
            Mutable accumulator for this (ticker, window_key, candidate_cik).
        score : float
            Evidence contribution after decay and weighting.
        evidence_id : str
            Deterministic ID of the evidence row (used to refresh the representative).
        filed_at : pd.Timestamp
            UTC-aware filing timestamp used to adjust first/last evidence dates.

        Returns
        -------
        None

        Notes
        -----
        - Adds `score` to the running total and increments `count` by 1.
        - If `score` exceeds `max_score`, replace the representative `evidence_id`.
        - Expands the envelope by updating `first_evidence_date` / `last_evidence_date`
          based on `filed_at` for downstream split checks.
        """

        current_candidate_data.score += score
        current_candidate_data.count += 1
        if score > current_candidate_data.max_score:
            current_candidate_data.max_score = score
            current_candidate_data.evidence_id = evidence_id
        if current_candidate_data.first_evidence_date > filed_at:
            current_candidate_data.first_evidence_date = filed_at
        elif current_candidate_data.last_evidence_date < filed_at:
            current_candidate_data.last_evidence_date = filed_at

    def scores(self, ticker: str, window_key: str) -> dict[str, CandidateData]:
        """
        Return the candidate aggregates for a (ticker, window_key), if any.

        Parameters
        ----------
        ticker : str
            Ticker key to query.
        window_key : str
            Canonical window identifier ('YYYY-MM-DD to YYYY-MM-DD').

        Returns
        -------
        dict[str, CandidateData]
            Mapping of candidate CIK → aggregate data; empty dict when no state exists.

        Notes
        -----
        - This is a read-only view over the internal accumulator.
        """

        scores_per_ticker = self.scores_and_counts.get(ticker)
        if scores_per_ticker is None:
            return {}
        return scores_per_ticker.get(window_key, {})

    def posterior_means(self, ticker: str, window_key: str) -> dict[str, float]:
        """
        Compute normalized posterior means over candidate CIKs for a (ticker, window).

        Parameters
        ----------
        ticker : str
            Ticker key to query.
        window_key : str
            Canonical window identifier (e.g., "YYYY-MM-DD to YYYY-MM-DD").

        Returns
        -------
        dict[str, float]
            Candidate CIK → posterior mean in [0, 1], summing to 1 across candidates.
            Empty dict if no evidence exists for the (ticker, window).

        Notes
        -----
        - Posterior mean for candidate c is (score_c + α) / Σ_j(score_j + α).
        - Only observed candidates receive mass (no “other” bucket).
        """

        score_and_counts_per_ticker = self.scores_and_counts.get(ticker)
        if score_and_counts_per_ticker is None:
            return {}
        score_and_counts = score_and_counts_per_ticker.get(window_key)
        if not score_and_counts:
            return {}
        denom = sum(sac.score + self.prior_pseudo_count for sac in score_and_counts.values())
        return {
            cik: (sac.score + self.prior_pseudo_count) / denom
            for cik, sac in score_and_counts.items()
        }

    def best_candidate(
        self, ticker: str, validity_window: str, exclude_cik: str | None = None
    ) -> BestCandidateResult | None:
        """
        Select the best CIK for a (ticker, window) using posterior mean and tie-breaks.

        Parameters
        ----------
        ticker : str
            Ticker key to query.
        validity_window : str
            Canonical window identifier ('YYYY-MM-DD to YYYY-MM-DD').
        exclude_cik : str | None
            Optional CIK to exclude from consideration (used when picking second-best).

        Returns
        -------
        BestCandidateResult | None
            The argmax candidate with its posterior and aggregate data, or None if no evidence.

        Notes
        -----
        - Tie-break order: (posterior_mean, max_score, -int(CIK)) for deterministic selection.
        """

        posterior_means: dict[str, float] = self.posterior_means(ticker, validity_window)
        if not posterior_means:
            return None
        max_scores: dict[str, float] = {
            cik: self.scores_and_counts[ticker][validity_window][cik].max_score
            for cik in posterior_means
        }
        best = max(
            posterior_means.keys(),
            key=lambda cik: (
                (posterior_means[cik], max_scores[cik], -int(cik))
                if cik != exclude_cik
                else (-1, -1, 0)
            ),
        )
        best_data: CandidateData = self.scores_and_counts[ticker][validity_window][best]
        if best_data is None:
            return None
        return BestCandidateResult(best, posterior_means[best], best_data)

    def handle_for_mid_window_split(
        self,
        ticker: str,
        best_candidate_result: BestCandidateResult,
        validity_window: ValidityWindow,
    ) -> MidWindowSplitData | None:
        """
        Evaluate and, if justified, compute a mid-window split against the runner-up.

        Parameters
        ----------
        ticker : str
            Ticker key.
        best_candidate_result : BestCandidateResult
            The current best candidate for the window.
        validity_window : ValidityWindow
            Original half-open [start, end) window for context.

        Returns
        -------
        MidWindowSplitData | None
            Split metadata (split_date, ordering, second-best) when conditions are met; else None.

        Notes
        -----
        - Split conditions are delegated to `check_split_window_conditions(...)`.
        - The split date is floored to UTC day and adjusted to preserve non-empty subwindows.
        """

        best_candidate: str = best_candidate_result.candidate_cik
        second_best_candidate_result = self.extract_second_best_candidate(
            ticker, validity_window, best_candidate
        )
        if second_best_candidate_result is None:
            return None
        best_candidate_data: CandidateData = best_candidate_result.candidate_data
        second_best_candidate_data: CandidateData = second_best_candidate_result.candidate_data
        if check_split_window_conditions(best_candidate_data, second_best_candidate_data):
            split_date, best_first = calculate_split_date(
                best_candidate_data, second_best_candidate_data
            )
            return MidWindowSplitData(
                split_date=split_date,
                second_best_candidate_result=second_best_candidate_result,
                best_first=best_first,
            )
        return None

    def extract_second_best_candidate(
        self, ticker: str, validity_window: ValidityWindow, best_candidate: str
    ) -> BestCandidateResult | None:
        """
        Retrieve the second-best candidate result for a window, if it exists.

        Parameters
        ----------
        ticker : str
            Ticker key.
        validity_window : ValidityWindow
            Half-open [start, end) window used to derive the `window_key`.
        best_candidate : str
            CIK of the best candidate to exclude from consideration.

        Returns
        -------
        BestCandidateResult | None
            Runner-up candidate with posterior and aggregates, or None if fewer than two candidates.

        Notes
        -----
        - Requires at least two distinct candidate CIKs in the window.
        """

        window_str: str = validity_window_to_str(validity_window)
        scores: dict[str, CandidateData] = self.scores(ticker, window_str)
        if len(scores.keys()) < 2:
            return None
        second_best_candidate_result: BestCandidateResult | None = self.best_candidate(
            ticker, window_str, exclude_cik=best_candidate
        )
        if second_best_candidate_result is None:
            return None
        return second_best_candidate_result
