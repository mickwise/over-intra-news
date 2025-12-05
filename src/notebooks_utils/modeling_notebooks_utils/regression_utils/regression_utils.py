from typing import List

import numpy as np
import pandas as pd
import sqlalchemy as sa
from rust_timeseries.duration_models import ACD  # pylint: disable=E0401:import-error

from notebooks_utils.data_notebooks_utils.general_data_notebooks_utils import (
    connect_with_sqlalchemy,
)
from notebooks_utils.modeling_notebooks_utils.regression_utils.regression_config import (
    FIRMS_TO_DROP,
)


def extract_topic_persistence_selection_data() -> pd.DataFrame:
    query: str = create_topic_persistence_selection_query()
    engine: sa.Engine = connect_with_sqlalchemy()
    with engine.connect() as connection:
        selection_df: pd.DataFrame = pd.read_sql(query, connection)
        selection_df["trading_day"] = pd.to_datetime(selection_df["trading_day"])
        firms_to_drop_mask: pd.Series = selection_df["cik"].isin(FIRMS_TO_DROP)
        selection_df = selection_df.loc[~firms_to_drop_mask].reset_index(drop=True)
    return selection_df


def create_topic_persistence_selection_query() -> str:
    """
    SQL Query Strategy:
    1. Valid News: Only articles that passed LDA cleaning.
    2. Universe: Full S&P 500 panel (Daily).
    3. Expansion: Cross join daily universe with (intraday, overnight) sessions.
    4. Join: Left join news to universe to preserve 'Phantom' (Zero-News) firms.
    """
    return """
    WITH valid_news_articles AS (
        SELECT 
            pna.article_id,
            pna.cik_list,
            pna.session,
            pna.trading_day
        FROM parsed_news_articles pna
        JOIN lda_documents ld ON pna.article_id = ld.article_id
        WHERE ld.included_in_training = TRUE 
    ),

    firm_day_counts AS (
        SELECT
            cik,
            session,
            trading_day,
            COUNT(*) AS daily_article_count
        FROM valid_news_articles p
        CROSS JOIN LATERAL unnest(p.cik_list) AS cik
        GROUP BY cik, session, trading_day
    ),

    durations AS (
        SELECT
            fdc.cik,
            fdc.session,
            fdc.trading_day,
            fdc.daily_article_count,
            (
                fdc.trading_day
                - LAG(fdc.trading_day) OVER (
                    PARTITION BY fdc.cik, fdc.session
                    ORDER BY fdc.trading_day
                )
            )::INT AS duration_days
        FROM firm_day_counts fdc
    ),

    universe AS (
        SELECT
            erp.cik,
            erp.trading_day,
            erp.log_market_cap,
            s.session
        FROM equity_regression_panel erp
        CROSS JOIN (VALUES ('intraday'), ('overnight')) AS s(session)
        WHERE erp.log_market_cap IS NOT NULL
    )

    SELECT
        u.cik,
        u.session,
        u.trading_day,
        COALESCE(d.daily_article_count, 0) AS daily_article_count,
        d.duration_days,
        u.log_market_cap
    FROM universe u
    LEFT JOIN durations d
        ON u.cik = d.cik
        AND u.session = d.session
        AND u.trading_day = d.trading_day
    ORDER BY u.cik, u.session, u.trading_day;
    """


def compute_firm_year_intensity(
    article_duration_df: pd.DataFrame,
    # Initial guess for [omega, alpha, beta] (roughly unit mean, high persistence)
    theta_0: np.ndarray = np.array([0.1, 0.3, 0.6], dtype=float),
) -> pd.DataFrame:
    per_event_rows: List[pd.DataFrame] = []

    for (cik, session), grp in article_duration_df.groupby(["cik", "session"]):
        cik_str: str = str(cik)
        session_str: str = str(session)
        # Sort is crucial for correct time-series reconstruction
        grp = grp.sort_values("trading_day")

        processed_df: pd.DataFrame = process_single_firm(cik_str, session_str, grp, theta_0)
        per_event_rows.append(processed_df)

    return handle_extracted_intensities(
        per_event_rows,
        article_duration_df,
    )


def process_single_firm(
    cik: str, session: str, grp: pd.DataFrame, theta_0: np.ndarray
) -> pd.DataFrame:
    event_mask: pd.Series = grp["duration_days"].notnull() & (grp["duration_days"] > 0)
    valid_events: pd.DataFrame = grp.loc[event_mask].copy()
    durations_raw: np.ndarray = valid_events["duration_days"].to_numpy(dtype=float)
    n_events: int = len(durations_raw)

    if n_events > 0:
        duration_mean = durations_raw.mean()
        durations_norm = durations_raw / duration_mean
    else:
        duration_mean = 1.0
        durations_norm = durations_raw

    if n_events >= 200:
        result = fit_tier_1_acd11(
            cik,
            session,
            valid_events,
            durations_norm,
            duration_mean,
            theta_0,
            tol_grad=1e-6,
            lbfgs_mem=20,
            line_searcher="MoreThuente",
        )
        if result is not None:
            return result

    if n_events >= 50:
        result = fit_tier_2_acd10(
            cik,
            session,
            valid_events,
            durations_norm,
            duration_mean,
            theta_0,
            tol_grad=1e-4,
            lbfgs_mem=50,
            line_searcher="HagerZhang",
        )
        if result is not None:
            return result

    return generate_tier_3_static(cik, session, grp)


def fit_tier_1_acd11(
    cik: str,
    session: str,
    valid_events: pd.DataFrame,
    durations_norm: np.ndarray,
    duration_mean: float,
    theta_0: np.ndarray,
    tol_grad: float,
    lbfgs_mem: int,
    line_searcher: str,
) -> pd.DataFrame | None:
    model: ACD = ACD(
        data_length=len(durations_norm),
        p=1,
        q=1,
        init="sample_mean",
        tol_grad=tol_grad,
        lbfgs_mem=lbfgs_mem,
        line_searcher=line_searcher,
    )
    model.fit(durations_norm, theta_0)

    if not model.results.converged:
        return None

    alpha: float = float(model.fitted_params.alpha[0])
    beta: float = float(model.fitted_params.beta[0])
    omega_norm: float = float(model.fitted_params.omega)

    if alpha + beta >= 0.99:
        return None

    omega_scaled = omega_norm * duration_mean

    return extract_intensities(
        cik,
        session,
        valid_events,
        durations_raw=valid_events["duration_days"].to_numpy(dtype=float),
        omega=omega_scaled,
        mass=alpha + beta,
        alpha=alpha,
        beta=beta,
    )


def fit_tier_2_acd10(
    cik: str,
    session: str,
    valid_events: pd.DataFrame,
    durations_norm: np.ndarray,
    duration_mean: float,
    theta_0: np.ndarray,
    tol_grad: float,
    lbfgs_mem: int,
    line_searcher: str,
) -> pd.DataFrame | None:
    model: ACD = ACD(
        data_length=len(durations_norm),
        p=0,
        q=1,
        init="sample_mean",
        tol_grad=tol_grad,
        lbfgs_mem=lbfgs_mem,
        line_searcher=line_searcher,
    )
    model.fit(durations_norm, theta_0[:2])

    if not model.results.converged:
        return None

    alpha: float = float(model.fitted_params.alpha[0])
    omega_norm: float = float(model.fitted_params.omega)

    # Stationarity Check: alpha < 1
    if alpha >= 0.99:
        return None

    omega_scaled = omega_norm * duration_mean

    return extract_intensities(
        cik,
        session,
        valid_events,
        durations_raw=valid_events["duration_days"].to_numpy(dtype=float),
        omega=omega_scaled,
        mass=alpha,
        alpha=alpha,
        beta=0.0,
    )


def generate_tier_3_static(cik: str, session: str, grp: pd.DataFrame) -> pd.DataFrame:
    tmp: pd.DataFrame = grp[["trading_day", "log_market_cap"]].copy()
    tmp["cik"] = cik
    tmp["session"] = session
    tmp["year"] = tmp["trading_day"].dt.year
    tmp["intensity"] = 0.0
    tmp["acd"] = False
    return tmp


def extract_intensities(
    cik: str,
    session: str,
    grp: pd.DataFrame,
    durations_raw: np.ndarray,
    omega: float,
    mass: float,
    alpha: float,
    beta: float,
) -> pd.DataFrame:
    psi: np.ndarray = np.empty_like(durations_raw)
    psi[0] = omega / (1.0 - mass)
    for t in range(1, len(durations_raw)):
        psi[t] = omega + alpha * durations_raw[t - 1] + beta * psi[t - 1]

    intensities: np.ndarray = 1.0 / psi

    tmp: pd.DataFrame = grp[["trading_day", "log_market_cap"]].copy()
    tmp["cik"] = cik
    tmp["session"] = session
    tmp["year"] = tmp["trading_day"].dt.year
    tmp["intensity"] = intensities
    tmp["acd"] = True
    return tmp


def handle_extracted_intensities(
    per_event_rows: List[pd.DataFrame],
    article_duration_df: pd.DataFrame,
) -> pd.DataFrame:
    if not per_event_rows:
        return pd.DataFrame(
            columns=[
                "cik",
                "session",
                "year",
                "intensity_per_day",
                "n_events_year",
                "n_durations_total",
                "log_market_cap",
            ]
        )

    event_level: pd.DataFrame = pd.concat(per_event_rows, ignore_index=True)

    firm_year: pd.DataFrame = event_level.groupby(["cik", "session", "year"], as_index=False).agg(
        yearly_intensity=("intensity", "mean"),
        log_market_cap=("log_market_cap", "mean"),
        n_events_year=("intensity", "size"),
    )

    n_durations: pd.DataFrame = (
        article_duration_df.groupby(["cik", "session"])["duration_days"]
        .count()
        .rename("n_durations_total")
        .reset_index()
    )

    firm_year = firm_year.merge(n_durations, on=["cik", "session"], how="left")
    return firm_year


def extract_topic_persistence_selection_data_by_firm(corpus_version: int = 1) -> pd.DataFrame:
    engine: sa.Engine = connect_with_sqlalchemy()
    cik_query = "SELECT DISTINCT cik FROM equity_regression_panel"
    with engine.connect() as connection:
        ciks = pd.read_sql(cik_query, connection)["cik"].tolist()
    print(f"Found {len(ciks)} unique firms. Starting micro-batch extraction...")
    dfs = []
    for i, target_cik in enumerate(ciks):
        if i % 50 == 0:
            print(f"Processing firm {i}/{len(ciks)}...")
        query = f"""
        SELECT
            '{target_cik}' as cik,
            pna.session,
            CAST(EXTRACT(YEAR FROM pna.trading_day) AS INTEGER) as year,
            late.topic_id,
            1 AS topic_exists
        FROM lda_documents ld
        JOIN parsed_news_articles pna 
            ON ld.article_id = pna.article_id
        JOIN lda_article_topic_exposure late 
            ON ld.article_id = late.article_id
        WHERE ld.corpus_version = {corpus_version}
          AND ld.included_in_training = TRUE
          AND late.topic_exposure > 0
          AND '{target_cik}' = ANY(pna.cik_list) -- GIN Index Filter
        GROUP BY 
            pna.session, 
            CAST(EXTRACT(YEAR FROM pna.trading_day) AS INTEGER), 
            late.topic_id
        """
        with engine.connect() as connection:
            chunk = pd.read_sql(query, connection)
            if not chunk.empty:
                dfs.append(chunk)
    if not dfs:
        return pd.DataFrame(columns=["cik", "session", "year", "topic_id", "topic_exists"])
    return pd.concat(dfs, ignore_index=True)


def build_topic_selection_panel(
    intensity_df: pd.DataFrame, corpus_version: int = 1
) -> pd.DataFrame:
    topic_hits = extract_topic_persistence_selection_data_by_firm(corpus_version)
    if topic_hits.empty:
        print("Warning: No topic hits found.")
        return pd.DataFrame()
    unique_topics = topic_hits["topic_id"].unique()
    skeleton = intensity_df.copy()
    skeleton["_cross_key"] = 1
    topics_frame = pd.DataFrame({"topic_id": unique_topics, "_cross_key": 1})
    full_panel = skeleton.merge(topics_frame, on="_cross_key").drop("_cross_key", axis=1)
    features_df = full_panel.merge(
        topic_hits, on=["cik", "session", "year", "topic_id"], how="left"
    )
    features_df["topic_exists"] = features_df["topic_exists"].fillna(0).astype(int)
    return features_df
