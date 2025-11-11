"""
Purpose
-------
Load EDGAR evidence rows and record run completion for (ticker × window × candidate_cik) jobs.

Key behaviors
-------------
- Streams `MappingEvidence` into `ticker_cik_evidence` via batched INSERT (execute_values).
- Serializes the window to a Postgres DATERANGE with half-open semantics `[start, end)`.
- Inserts a single row into `edgar_run_registry` to mark the (ticker, validity_window,
  candidate_cik) triple as completed.

Conventions
-----------
- Evidence batch size is driven by the upstream `load_into_table` helper.
- `raw_record` is written as JSONB using `psycopg2.extras.Json`.
- `validity_window` is stored as `[)` using `psycopg2.extras.DateRange`.

Downstream usage
----------------
- Call `persist_collected_data(run_id, ticker, candidate_cik, validity_window,
  start_time, logger, collected_evidence)` after harvesting completes for a triple.
- `persist_collected_data(...)` will:
    * write `collected_evidence` into `ticker_cik_evidence`, and
    * insert a completion row into `edgar_run_registry` for
      (ticker, validity_window, candidate_cik) in the same transaction.
"""

from typing import Iterator, List

import pandas as pd
import psycopg2

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.records.evidence_record import MappingEvidence
from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindow
from infra.utils.db_utils import connect_to_db, load_into_table


def persist_collected_data(
    run_id: str,
    ticker: str,
    validity_window: ValidityWindow,
    candidate_cik: str,
    start_time: pd.Timestamp,
    logger: InfraLogger,
    collected_evidence: List[MappingEvidence],
) -> None:
    """
    Persist all evidence for a single (ticker × window × candidate_cik) and record
    completion in the run registry.

    Parameters
    ----------
    run_id : str
        Unique identifier for the current run (UUID4).
    ticker : str
        Canonicalized ticker for which evidence was collected.
    candidate_cik : str
        10-digit CIK used as the EDGAR query identifier for this harvest.
    validity_window : ValidityWindow
        Half-open [start_utc, end_utc) window used during harvesting.
    start_time : pandas.Timestamp
        UTC timestamp marking the start of the collection for this
        (ticker, validity_window, candidate_cik) triple.
    logger : InfraLogger
        Structured logger for DEBUG/INFO events.
    collected_evidence : list[MappingEvidence]
        Evidence records gathered for this window and candidate.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Propagates any database errors during insert/commit; callers should rely on the
        connection context manager to roll back on exceptions.

    Notes
    -----
    - Evidence rows are inserted first; the run registry row is inserted only on success.
    - Both inserts are performed within a single transaction; a failure in the second
      step will roll back the evidence insert.
    - The run registry insert relies on the (ticker, validity_window, candidate_cik)
      uniqueness to make the operation idempotent across reruns.
    """

    with connect_to_db() as conn:
        logger.debug(
            "persist_start",
            context={"stage": "edgar_search", "ticker": ticker, "run_id": run_id},
        )
        load_evidence_table(conn, ticker, validity_window, collected_evidence)
        logger.debug(
            "persist_evidence_complete",
            context={"stage": "edgar_search", "ticker": ticker, "run_id": run_id},
        )
        run_registry_query: str = generate_run_registry_query()
        with conn.cursor() as cursor:
            cursor.execute(
                run_registry_query,
                (
                    ticker,
                    psycopg2.extras.DateRange(
                        validity_window[0].date(), validity_window[1].date(), "[)"
                    ),
                    candidate_cik,
                    run_id,
                    start_time,
                ),
            )
        conn.commit()
        logger.debug(
            "persist_run_registry_complete",
            context={"stage": "edgar_search", "ticker": ticker, "run_id": run_id},
        )


def load_evidence_table(
    conn: psycopg2.extensions.connection,
    ticker: str,
    validity_window: ValidityWindow,
    collected_evidence: List[MappingEvidence],
) -> None:
    """
    Stream a list of `MappingEvidence` into `ticker_cik_evidence` using batched INSERTs.

    Parameters
    ----------
    conn : psycopg2.extensions.connection
        Open PostgreSQL connection; the caller owns commit/rollback/close.
    ticker : str
        Canonical ticker; repeated for each row to avoid recomputation at generation time.
    validity_window : ValidityWindow
        Half-open `[start, end)` window persisted as a DATERANGE.
    collected_evidence : list[MappingEvidence]
        Evidence objects to be inserted.

    Returns
    -------
    None

    Notes
    -----
    - Delegates batching and execution to `infra.utils.db_utils.load_into_table`.
    - Uses `generate_evidence_query()` for the `INSERT ... VALUES %s` template.
    - The final short batch is flushed by `load_into_table`.
    """

    query: str = generate_evidence_query()
    row_generator: Iterator[tuple] = create_row_generator_evidence(
        ticker, validity_window, collected_evidence
    )
    load_into_table(conn, row_generator, query)


def generate_evidence_query() -> str:
    """
    Build the batched INSERT template for `ticker_cik_evidence`.

    Returns
    -------
    str
        SQL string compatible with `psycopg2.extras.execute_values`, including a single
        `%s` placeholder for the VALUES block.

    Notes
    -----
    - Conflict policy is `ON CONFLICT (evidence_id) DO NOTHING` to ensure idempotency.
    """

    return """
    INSERT INTO ticker_cik_evidence (
        ticker,
        candidate_cik,
        evidence_id,
        validity_window,
        filed_at,
        accession_num,
        form_type,
        source,
        raw_record,
        company_name,
        items_8k,
        items_descriptions_8k
    ) VALUES %s
    ON CONFLICT (evidence_id) DO NOTHING;
    """


def create_row_generator_evidence(
    ticker: str,
    validity_window: ValidityWindow,
    collected_evidence: List[MappingEvidence],
) -> Iterator[tuple]:
    """
    Yield `ticker_cik_evidence` value tuples for `execute_values`.

    Parameters
    ----------
    ticker : str
        Canonical ticker to repeat across rows.
    validity_window : ValidityWindow
        Half-open `[start, end)` window stored as a DATERANGE with bounds `[)`.
    collected_evidence : list[MappingEvidence]
        Evidence objects to convert into row tuples.

    Yields
    ------
    tuple
        `(ticker, candidate_cik, evidence_id, validity_window, filed_at,
          accession_num, form_type, source, raw_record_jsonb, company_name,
          items_8k, items_descriptions_8k)`

    Notes
    -----
    - `raw_record` is wrapped with `psycopg2.extras.Json(...)` for JSONB
      insertion.
    - Dates are persisted as a `psycopg2.extras.DateRange(lower, upper, "[)")`.
    - `items_8k` and `items_descriptions_8k` are passed through as Python lists
      or None and stored as TEXT[] columns.
    """

    date_validity_window: psycopg2.extras.DateRange = psycopg2.extras.DateRange(
        validity_window[0].date(), validity_window[1].date(), "[)"
    )
    for evidence in collected_evidence:
        yield (
            ticker,
            evidence.candidate_cik,
            evidence.evidence_id,
            date_validity_window,
            evidence.filed_at,
            evidence.accession_num,
            evidence.form_type,
            evidence.source,
            psycopg2.extras.Json(evidence.raw_record),
            evidence.company_name,
            evidence.items_8k,
            evidence.items_descriptions_8k,
        )


def generate_run_registry_query() -> str:
    """
    Build the parameterized INSERT for `edgar_run_registry`.

    Returns
    -------
    str
        SQL with `%s` placeholders for `(ticker, validity_window, candidate_cik,
        run_id, start_time)`.

    Notes
    -----
    - Uses `ON CONFLICT (ticker, validity_window, candidate_cik) DO NOTHING` so
      repeated calls for the same triple are idempotent.
    """
    return """
        INSERT INTO edgar_run_registry (
            ticker,
            validity_window,
            candidate_cik,
            run_id,
            start_time
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (ticker, validity_window, candidate_cik) DO NOTHING;
        """
