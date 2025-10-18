from typing import List

import pandas as pd

from infra.seeds.seed_firm_characteristics.edgar_search.edgar_search_utils import (
    CollectedEvidence,
    extract_evidence_by_id,
)
from infra.seeds.seed_firm_characteristics.records.table_records import (
    MappingEvidence,
    MappingRecord,
    ValidityWindow,
    validity_window_to_str,
)
from infra.seeds.seed_firm_characteristics.scoring.confidence_scores import ConfidenceScores
from infra.seeds.seed_firm_characteristics.scoring.scoring_utils import (
    BestCandidateResult,
    MidWindowSplitData,
    confidence_to_label,
)


def build_mapping_records_list(
    ticker_validity_windows: dict[str, List[ValidityWindow]],
    collected_evidence: CollectedEvidence,
    confidence_scores: ConfidenceScores,
) -> List[MappingRecord]:
    mapping_records: List[MappingRecord] = []
    for ticker, validity_windows in ticker_validity_windows.items():
        for validity_window in validity_windows:
            window_key: str = validity_window_to_str(validity_window)
            best_candidate_result: BestCandidateResult | None = confidence_scores.best_candidate(
                ticker, window_key
            )
            if best_candidate_result is None:
                continue
            best_evidence_id: str = best_candidate_result.candidate_data.evidence_id
            best_evidence: MappingEvidence | None = extract_evidence_by_id(
                ticker, window_key, best_evidence_id, collected_evidence
            )
            if best_evidence is None:
                continue
            mid_window_split_data: MidWindowSplitData | None = (
                confidence_scores.handle_for_mid_window_split(
                    ticker, best_candidate_result, validity_window
                )
            )
            if mid_window_split_data is not None:
                handle_split_records(
                    ticker,
                    validity_window,
                    collected_evidence,
                    best_candidate_result,
                    best_evidence,
                    mid_window_split_data,
                    mapping_records,
                )
            else:
                append_single_mapping_record(
                    ticker, validity_window, best_evidence, best_candidate_result, mapping_records
                )
    return mapping_records


def handle_split_records(
    ticker: str,
    validity_window: ValidityWindow,
    collected_evidence: CollectedEvidence,
    best_candidate_result: BestCandidateResult,
    best_evidence: MappingEvidence,
    mid_window_split_data: MidWindowSplitData,
    mapping_records: List[MappingRecord],
) -> None:
    second_best_evidence_id: str = (
        mid_window_split_data.second_best_candidate_result.candidate_data.evidence_id
    )
    second_best_evidence: MappingEvidence | None = extract_evidence_by_id(
        ticker, validity_window_to_str(validity_window), second_best_evidence_id, collected_evidence
    )
    if second_best_evidence is None:
        return None
    first_window, second_window = build_split_validity_windows(
        validity_window, mid_window_split_data.split_date
    )
    best_first: bool = mid_window_split_data.best_first
    if best_first:
        append_single_mapping_record(
            ticker, first_window, best_evidence, best_candidate_result, mapping_records
        )
        append_single_mapping_record(
            ticker,
            second_window,
            second_best_evidence,
            mid_window_split_data.second_best_candidate_result,
            mapping_records,
        )
    else:
        append_single_mapping_record(
            ticker,
            first_window,
            second_best_evidence,
            mid_window_split_data.second_best_candidate_result,
            mapping_records,
        )
        append_single_mapping_record(
            ticker, second_window, best_evidence, best_candidate_result, mapping_records
        )


def build_split_validity_windows(
    validity_window: ValidityWindow, split_date: pd.Timestamp
) -> tuple[ValidityWindow, ValidityWindow]:
    first_window: ValidityWindow = (validity_window[0], split_date)
    second_window: ValidityWindow = (split_date, validity_window[1])
    return first_window, second_window


def append_single_mapping_record(
    ticker: str,
    validity_window: ValidityWindow,
    evidence: MappingEvidence,
    candidate_result: BestCandidateResult,
    mapping_records: List[MappingRecord],
) -> None:
    record: MappingRecord = MappingRecord(
        ticker=ticker,
        cik=candidate_result.candidate_cik,
        filed_at=evidence.filed_at,
        validity_window=validity_window,
        evidence_type=evidence.form_type,
        evidence_id=evidence.evidence_id,
        accession_num=evidence.accession_num,
        source=evidence.source,
        confidence=confidence_to_label(candidate_result.posterior_mean),
        confidence_score=candidate_result.posterior_mean,
    )
    mapping_records.append(record)
