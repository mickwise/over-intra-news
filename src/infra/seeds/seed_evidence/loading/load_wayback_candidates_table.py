"""
Purpose
-------
Persist Wayback Machine–derived ticker–CIK candidates into the warehouse.
Provides a thin, batch-oriented loading layer that turns in-memory
WayBackCandidate objects into rows for the `wayback_cik_candidates` table.

Key behaviors
-------------
- Generate the parametrized INSERT .. ON CONFLICT DO NOTHING statement
  used to load Wayback candidates.
- Convert a list of WayBackCandidate instances into tuples compatible
  with `psycopg2.extras.execute_values`, including a DATERANGE
  validity_window.
- Open a Postgres connection via `connect_to_db()` and bulk-insert all
  candidates using `load_into_table()`.

Conventions
-----------
- WayBackCandidate carries at least:
  `(ticker, validity_window, candidate_cik, first_seen, last_seen,
    first_seen_url, last_seen_url)`.
- validity_window is stored as a `psycopg2.extras.DateRange` with
  half-open `[start, end)` bounds in the database.
- Idempotency (re-running the same load safely) is enforced by a
  unique constraint on `wayback_cik_candidates` and the
  `ON CONFLICT DO NOTHING` clause.

Downstream usage
----------------
- Use `load_wayback_candidates(...)` at the end of the Wayback seeding
  pipeline to materialize all discovered candidates into the warehouse.
- Use `generate_wayback_candidate_query()` and
  `create_wayback_candidate_row_generator()` directly when composing
  custom loaders that share the same table contract but control
  connection lifecycle or batching strategy explicitly.
"""

from typing import Iterable, List

import psycopg2.extras

from infra.seeds.seed_evidence.wayback.wayback_config import WayBackCandidate
from infra.utils.db_utils import connect_to_db, load_into_table


def load_wayback_candidates(candidates: List[WayBackCandidate]) -> None:
    """
    Insert a batch of Wayback-derived CIK candidates into `wayback_candidates`.

    Parameters
    ----------
    candidates : list[WayBackCandidate]
        In-memory candidate objects produced by the Wayback seeding pipeline.
        Each instance carries `(ticker, validity_window, candidate_cik,
        first_seen, last_seen, first_seen_url, last_seen_url)`.

    Returns
    -------
    None
        Inserts rows as a side effect; does not return a value.

    Raises
    ------
    psycopg2.Error
        Propagated if the INSERT fails or the connection cannot be established.
    BaseException
        Propagated from `connect_to_db(...)` or `load_into_table(...)` on
        unexpected driver- or environment-level failures.

    Notes
    -----
    - Uses `load_into_table(...)` to batch INSERTs via `execute_values`.
    - Commits the transaction before leaving the connection context.
    - Idempotency is handled by the `ON CONFLICT` clause in
      `generate_wayback_candidate_query()`.
    """

    query: str = generate_wayback_candidate_query()
    generator: Iterable[tuple] = create_wayback_candidate_row_generator(candidates)
    with connect_to_db() as conn:
        load_into_table(conn, generator, query)


def generate_wayback_candidate_query() -> str:
    """
    Build the batched INSERT template for `wayback_candidates`.

    Returns
    -------
    str
        SQL string compatible with `psycopg2.extras.execute_values`, including
        a single `%s` placeholder for the VALUES block.

    Notes
    -----
    - The conflict policy is `ON CONFLICT DO NOTHING`, relying on a unique
      constraint defined on the table (for example,
      `(ticker, validity_window, candidate_cik)`).
    """

    return """
    INSERT INTO wayback_cik_candidates (
        ticker,
        validity_window,
        candidate_cik,
        first_seen_at,
        last_seen_at,
        first_seen_url,
        last_seen_url
    ) VALUES %s 
    ON CONFLICT DO NOTHING;
    """


def create_wayback_candidate_row_generator(
    candidates: List[WayBackCandidate],
) -> Iterable[tuple]:
    """
    Yield value tuples for inserting into `wayback_candidates`.

    Parameters
    ----------
    candidates : list[WayBackCandidate]
        Candidate objects to convert into row tuples.

    Yields
    ------
    tuple
        `(ticker, validity_window_daterange, candidate_cik,
          first_seen_at, last_seen_at, first_seen_url, last_seen_url)`
        where `validity_window_daterange` is a `psycopg2.extras.DateRange`
        with `[start, end)` bounds.

    Notes
    -----
    - The `ValidityWindow` on each candidate is assumed to be a pair of
      tz-aware `pandas.Timestamp` objects; only the `.date()` component is
      persisted into the DATERANGE.
    """

    for candidate in candidates:
        validity_window_daterange: psycopg2.extras.DateRange = psycopg2.extras.DateRange(
            candidate.validity_window[0].date(), candidate.validity_window[1].date(), "[)"
        )
        yield (
            candidate.ticker,
            validity_window_daterange,
            candidate.candidate_cik,
            candidate.first_seen,
            candidate.last_seen,
            candidate.first_seen_url,
            candidate.last_seen_url,
        )
