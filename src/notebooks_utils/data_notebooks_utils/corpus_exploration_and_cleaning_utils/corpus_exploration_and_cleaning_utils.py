import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sa

# fmt: off
from notebooks_utils.data_notebooks_utils.\
    corpus_exploration_and_cleaning_utils.corpus_exploration_and_cleaning_config import (
    NOISY_PREFIXES,
    NOISY_SUBSTRINGS,
    STRONG_ENGLISH_CONFIDENCE_THRESHOLD,
)

from infra.logging.infra_logger import InfraLogger
from notebooks_utils.data_notebooks_utils.general_data_notebooks_utils import (
    connect_with_sqlalchemy,
)

# fmt: on


def sample_corpus_per_day(start_date: dt.date, end_date: dt.date, logger: InfraLogger) -> set[str]:
    trading_calendar_query: str = """
    SELECT trading_day
    FROM trading_calendar
    WHERE trading_day BETWEEN %s AND %s
    AND is_trading_day = TRUE;
    """

    engine: sa.Engine = connect_with_sqlalchemy()
    article_samples: set[str] = set()
    with engine.connect() as conn:
        trading_days_df: pd.DataFrame = pd.read_sql(
            trading_calendar_query, conn, params=(start_date, end_date)
        )
        for day in trading_days_df["trading_day"]:
            for session in ["overnight", "intraday"]:
                logger.debug(
                    event="sample_corpus_per_day", context={"day": day, "session": session}
                )
                corpus_query: str = """
                SELECT 
                article_id,
                language_confidence,
                full_text
                FROM parsed_news_articles
                WHERE trading_day = %s AND session = %s;
                """
                daily_sample_df: pd.DataFrame = pd.read_sql(
                    corpus_query, conn, params=(day, session)
                )
                article_samples.update(sample_per_day_session(daily_sample_df))
                logger.debug(
                    event="sample_corpus_per_day",
                    context={
                        "day": day,
                        "session": session,
                        "sampled_article_count": len(article_samples),
                    },
                )
    return article_samples


def sample_per_day_session(daily_sample_df: pd.DataFrame) -> set[str]:
    boilerplate_prefix_mask: pd.Series = daily_sample_df["full_text"].str.startswith(NOISY_PREFIXES)
    boilerplate_substring_mask: pd.Series = daily_sample_df["full_text"].str.contains(
        "|".join(NOISY_SUBSTRINGS)
    )
    strong_english_mask: pd.Series = (
        daily_sample_df["language_confidence"] >= STRONG_ENGLISH_CONFIDENCE_THRESHOLD
    )
    clean_articles_df: pd.DataFrame = daily_sample_df.loc[
        ~boilerplate_prefix_mask & ~boilerplate_substring_mask & strong_english_mask
    ]
    return set(clean_articles_df["article_id"].tolist())


def plot_article_temporal_and_cik_coverage(sample_df: pd.DataFrame) -> None:
    sample_df_cp = sample_df.copy()
    sample_df_cp["trading_day"] = pd.to_datetime(sample_df_cp["trading_day"])

    daily_counts = (
        sample_df_cp.groupby(["trading_day", "session"])["article_id"]
        .count()
        .reset_index(name="n_articles")
    )

    overnight_counts = daily_counts[daily_counts["session"] == "overnight"]
    intraday_counts = daily_counts[daily_counts["session"] == "intraday"]
    both_counts = (
        daily_counts.groupby("trading_day")["n_articles"].sum().reset_index(name="n_articles")
    )
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Overnight
    axes1[0].plot(
        overnight_counts["trading_day"],
        overnight_counts["n_articles"],
    )
    axes1[0].set_title("Articles per day – Overnight")
    axes1[0].set_ylabel("# articles")
    axes1[0].grid(True, alpha=0.3)

    # Intraday
    axes1[1].plot(
        intraday_counts["trading_day"],
        intraday_counts["n_articles"],
    )
    axes1[1].set_title("Articles per day – Intraday")
    axes1[1].set_ylabel("# articles")
    axes1[1].grid(True, alpha=0.3)

    # Both sessions combined
    axes1[2].plot(
        both_counts["trading_day"],
        both_counts["n_articles"],
    )
    axes1[2].set_title("Articles per day – Both sessions")
    axes1[2].set_xlabel("Trading day")
    axes1[2].set_ylabel("# articles")
    axes1[2].grid(True, alpha=0.3)

    # Make the x-axis a bit more readable
    axes1[2].xaxis.set_major_locator(mdates.YearLocator())
    axes1[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig1.autofmt_xdate()
    fig1.tight_layout()

    exploded = (
        sample_df_cp[["article_id", "trading_day", "cik_list"]]
        .explode("cik_list")
        .dropna(subset=["cik_list"])
        .rename(columns={"cik_list": "cik"})
    )
    exploded["trading_day"] = pd.to_datetime(exploded["trading_day"])
    exploded["year"] = exploded["trading_day"].dt.year
    exploded["year_month"] = exploded["trading_day"].dt.to_period("M")

    articles_per_cik = exploded.groupby("cik")["article_id"].nunique()
    days_per_cik = exploded.groupby("cik")["trading_day"].nunique()
    months_per_cik = exploded.groupby("cik")["year_month"].nunique()
    years_per_cik = exploded.groupby("cik")["year"].nunique()

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # 2.1 Articles per CIK
    axes2[0, 0].hist(articles_per_cik.values, bins=50)
    axes2[0, 0].set_title("Articles per CIK")
    axes2[0, 0].set_xlabel("# articles mentioning CIK")
    axes2[0, 0].set_ylabel("# CIKs")

    # 2.2 Unique trading days per CIK
    axes2[0, 1].hist(days_per_cik.values, bins=50)
    axes2[0, 1].set_title("Unique trading days per CIK")
    axes2[0, 1].set_xlabel("# trading days with ≥1 article")
    axes2[0, 1].set_ylabel("# CIKs")

    # 2.3 Unique months per CIK
    axes2[1, 0].hist(months_per_cik.values, bins=50)
    axes2[1, 0].set_title("Unique months per CIK")
    axes2[1, 0].set_xlabel("# months with ≥1 article")
    axes2[1, 0].set_ylabel("# CIKs")

    # 2.4 Unique years per CIK
    axes2[1, 1].hist(years_per_cik.values, bins=range(1, years_per_cik.max() + 2))
    axes2[1, 1].set_title("Unique years per CIK")
    axes2[1, 1].set_xlabel("# years with ≥1 article")
    axes2[1, 1].set_ylabel("# CIKs")

    fig2.tight_layout()

    plt.show()


def compute_coverage(
    filtered_sample_df: pd.DataFrame,
    cik_window_df: pd.DataFrame,
) -> pd.DataFrame:
    sample_df_cp: pd.DataFrame = filtered_sample_df.copy()
    sample_df_cp["trading_day"] = pd.to_datetime(sample_df_cp["trading_day"]).dt.date

    exploded: pd.DataFrame = sample_df_cp.explode("cik_list").rename(columns={"cik_list": "cik"})

    all_sample_trading_days: list[dt.date] = sorted(set(sample_df_cp["trading_day"]))
    coverage_rows: list[dict] = []

    for cik, group in exploded.groupby("cik"):
        cik_rows: pd.DataFrame = cik_window_df[cik_window_df["cik"] == cik]

        all_days_set: set[dt.date] = set()
        article_days_set: set[dt.date] = set()

        cik_article_days: set[dt.date] = set(group["trading_day"].unique())

        for _, row in cik_rows.iterrows():
            validity_window = row["validity_window"]
            window_start: dt.date = validity_window.lower
            window_end: dt.date = validity_window.upper - dt.timedelta(days=1)

            window_trading_days: list[dt.date] = [
                trading_day
                for trading_day in all_sample_trading_days
                if window_start <= trading_day <= window_end
            ]
            all_days_set.update(window_trading_days)

            window_article_days: list[dt.date] = [
                trading_day
                for trading_day in cik_article_days
                if window_start <= trading_day <= window_end
            ]
            article_days_set.update(window_article_days)

        all_days: list[dt.date] = sorted(all_days_set)
        article_days: list[dt.date] = sorted(article_days_set)

        all_years = {d.year for d in all_days}
        article_years = {d.year for d in article_days}
        total_years = len(all_years)
        covered_years = len(all_years & article_years)
        year_cov_ratio = covered_years / total_years if total_years > 0 else 0.0

        all_months = {(d.year, d.month) for d in all_days}
        article_months = {(d.year, d.month) for d in article_days}
        total_months = len(all_months)
        covered_months = len(all_months & article_months)
        month_cov_ratio = covered_months / total_months if total_months > 0 else 0.0

        all_day_set = set(all_days)
        article_day_set = set(article_days)
        total_days = len(all_day_set)
        covered_days = len(all_day_set & article_day_set)
        day_cov_ratio = covered_days / total_days if total_days > 0 else 0.0

        coverage_rows.append(
            {
                "cik": cik,
                "year_coverage_ratio": year_cov_ratio,
                "month_coverage_ratio": month_cov_ratio,
                "day_coverage_ratio": day_cov_ratio,
                "total_years_in_window": total_years,
                "total_months_in_window": total_months,
                "total_days_in_window": total_days,
            }
        )

    coverage_df = pd.DataFrame(coverage_rows)
    return coverage_df


def plot_window_normalized_coverage(coverage_df: pd.DataFrame) -> None:
    _, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    # Year coverage
    axes[0].hist(coverage_df["year_coverage_ratio"], bins=20)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_title("Per-CIK year coverage ratio (relative to validity window)")
    axes[0].set_xlabel("Fraction of years in validity window with ≥1 article")
    axes[0].set_ylabel("# CIKs")

    # Month coverage
    axes[1].hist(coverage_df["month_coverage_ratio"], bins=20)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_title("Per-CIK month coverage ratio (relative to validity window)")
    axes[1].set_xlabel("Fraction of months in validity window with ≥1 article")
    axes[1].set_ylabel("# CIKs")

    # Day coverage
    axes[2].hist(coverage_df["day_coverage_ratio"], bins=20)
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_title("Per-CIK day coverage ratio (relative to validity window)")
    axes[2].set_xlabel("Fraction of days in validity window with ≥1 article")
    axes[2].set_ylabel("# CIKs")

    plt.tight_layout()
    plt.show()
