from typing import Iterator, List

import psycopg2
import psycopg2.extras

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_firm_characteristics.edgar_search.edgar_search_core import (
    CollectedEvidence,
    PotentialNames,
)
from infra.seeds.seed_firm_characteristics.loading.load_tables_utils import (
    build_mapping_records_list,
)
from infra.seeds.seed_firm_characteristics.records.table_records import (
    MappingRecord,
    ValidityWindows,
    str_to_validity_window,
)
from infra.seeds.seed_firm_characteristics.scoring.confidence_scores import ConfidenceScores
from infra.utils.db_utils import connect_to_db, load_into_table


def load_tables(
    ticker_validity_windows: dict[str, ValidityWindows],
    collected_evidence: CollectedEvidence,
    confidence_scores: ConfidenceScores,
    potential_names: PotentialNames,
    logger: InfraLogger,
) -> None:
    with connect_to_db() as conn:
        logger.info("Loading ticker_cik_evidence table")
        load_evidence_table(conn, collected_evidence)
        logger.info("Finished loading ticker_cik_evidence table")
        mapping_records: List[MappingRecord] = build_mapping_records_list(
            ticker_validity_windows, collected_evidence, confidence_scores
        )
        logger.info("Loading ticker_cik_mapping table")
        load_mapping_table(conn, mapping_records)
        logger.info("Finished loading ticker_cik_mapping table")


def load_evidence_table(
    conn: psycopg2.extensions.connection, collected_evidence: CollectedEvidence
) -> None:
    query: str = generate_evidence_query()
    row_generator = create_row_generator_evidence(collected_evidence)
    load_into_table(conn, row_generator, query)


def generate_evidence_query() -> str:
    return """
    INSERT INTO firm_characteristics.ticker_cik_evidence (
        ticker,
        candidate_cik,
        evidence_id,
        validity_window,
        filed_at,
        accession_num,
        form_type,
        source,
        raw_record
    ) VALUES %s
    ON CONFLICT (evidence_id) DO NOTHING;
    """


def create_row_generator_evidence(collected_evidence: CollectedEvidence) -> Iterator:
    for ticker, windows_dict in collected_evidence.items():
        for window, evidence_dict in windows_dict.items():
            start_date, end_date = str_to_validity_window(window)
            date_validity_window: psycopg2.extras.DateRange = psycopg2.extras.DateRange(
                start_date.date(), end_date.date(), "[)"
            )
            for evidence_id, evidence in evidence_dict.items():
                yield (
                    ticker,
                    evidence.candidate_cik,
                    evidence_id,
                    date_validity_window,
                    evidence.filed_at,
                    evidence.accession_num,
                    evidence.form_type,
                    evidence.source,
                    evidence.raw_record,
                )


def load_mapping_table(
    conn: psycopg2.extensions.connection, mapping_records: List[MappingRecord]
) -> None:
    query: str = generate_mapping_query()
    row_generator = create_row_generator_mapping(mapping_records)
    load_into_table(conn, row_generator, query)


def generate_mapping_query() -> str:
    return """
    INSERT INTO firm_characteristics.ticker_cik_mapping (
        ticker,
        cik,
        validity_window,
        evidence_type,
        filed_at,
        source,
        accession_num,
        confidence,
        confidence_score,
        evidence_id
    ) VALUES %s
    ON CONFLICT (ticker, cik, validity_window) DO NOTHING;
    """


def create_row_generator_mapping(mapping_records: List[MappingRecord]) -> Iterator:
    for record in mapping_records:
        start_date, end_date = record.validity_window
        date_validity_window: psycopg2.extras.DateRange = psycopg2.extras.DateRange(
            start_date.date(), end_date.date(), "[)"
        )
        yield (
            record.ticker,
            record.cik,
            date_validity_window,
            record.evidence_type,
            record.filed_at,
            record.source,
            record.accession_num,
            record.confidence,
            record.confidence_score,
            record.evidence_id,
        )


def generate_security_master_query() -> str:
    return """
    INSERT SELECT DISTINCT cik FROM 0002_ticker_cik_mapping
    ON CONFLICT (cik) DO NOTHING;
    """
