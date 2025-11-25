"""
Purpose
-------
Provide diagnostic visualization utilities for assessing the temporal and
cross-sectional structure of the sampled news corpus. The plots in this module
summarize article volumes, CIK participation, and token/document frequency
distributions to ensure that the cleaned corpus behaves sensibly before LDA
construction.

Key behaviors
-------------
- Plot daily article counts across overnight, intraday, and combined sessions.
- Measure and visualize CIK coverage relative to validity windows.
- Produce frequency-of-frequency histograms for both global term counts and
  document frequencies.
- Operate purely in-memory for exploratory analysis without modifying inputs.

Conventions
-----------
- Assumes trading_day values are parsable into pandas date objects.
- Expects cik_list fields to contain lists of identifiers per article.
- All figures are produced with matplotlib and shown immediately; no files are
  written.
- All transformations on input DataFrames are performed on local copies to avoid
  mutating upstream data.

Downstream usage
----------------
Call these utilities from notebooks or ad-hoc diagnostics to validate corpus
health before committing to vocabulary construction or LDA training. The module
should not be imported by production pipelines; it is intended exclusively for
visual inspection and sanity checks.
"""

import datetime as dt
from typing import Counter

import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt


def plot_article_temporal_and_cik_coverage(sample_df: pd.DataFrame) -> None:
    """
    Plot three daily article-count time series (overnight, intraday, combined) and
    four distributional histograms summarizing article and CIK temporal coverage.

    Parameters
    ----------
    sample_df : pandas.DataFrame
        Must contain ['article_id', 'trading_day', 'session', 'cik_list'] with
        cik_list as a list per row.

    Returns
    -------
    None
        Displays matplotlib figures.

    Notes
    -----
    - Used only for diagnostics and exploratory analysis.
    - Does not modify data; all transformations are local copies.
    """

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
    """
    Compute coverage ratios (days, months, years) for each CIK relative to its
    validity windows and the sample’s trading-day universe.

    Parameters
    ----------
    filtered_sample_df : pandas.DataFrame
        DataFrame of sampled articles after gating, containing trading_day and
        cik_list.
    cik_window_df : pandas.DataFrame
        Contains ['cik', 'validity_window'], where validity_window is a Postgres
        daterange giving when the CIK was active.

    Returns
    -------
    pandas.DataFrame
        One row per CIK with ratios for year/month/day coverage and absolute counts.

    Notes
    -----
    - Validity windows are treated as [start, end) half-open.
    - Trading days come from the filtered_sample_df universe only.
    """

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
    """
    Plot histograms of per-CIK coverage ratios (year, month, day) derived from
    compute_coverage.

    Parameters
    ----------
    coverage_df : pandas.DataFrame
        Output of compute_coverage.

    Returns
    -------
    None
        Displays three matplotlib histograms.

    Notes
    -----
    - purely diagnostic; does not mutate coverage_df.
    """

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


def plot_token_and_doc_frequency(
    token_counts: Counter[str],
    doc_counts: Counter[str],
    max_token_freq: int = 1000,
    max_doc_freq: int = 1000,
    log_y: bool = True,
) -> None:
    """
    Plot frequency-of-frequency histograms for (1) global token frequency and
    (2) token document frequency.

    Parameters
    ----------
    token_counts : Counter[str]
        Global term frequencies.
    doc_counts : Counter[str]
        Document frequencies.
    max_token_freq : int, optional
        Cap for token frequency histogram.
    max_doc_freq : int, optional
        Cap for document frequency histogram.
    log_y : bool
        Whether to log-scale y-axis.

    Returns
    -------
    None
        Displays matplotlib plots.

    Notes
    -----
    - Used for corpus diagnostics; does not modify counter inputs.
    """

    token_freq_values = list(token_counts.values())
    if max_token_freq is not None:
        token_freq_values = [v for v in token_freq_values if v <= max_token_freq]

    token_freq_of_freqs = Counter(token_freq_values)
    token_k = sorted(token_freq_of_freqs.keys())
    token_counts_of_counts = [token_freq_of_freqs[k] for k in token_k]

    doc_freq_values = list(doc_counts.values())
    if max_doc_freq is not None:
        doc_freq_values = [v for v in doc_freq_values if v <= max_doc_freq]

    doc_freq_of_freqs = Counter(doc_freq_values)
    doc_k = sorted(doc_freq_of_freqs.keys())
    doc_counts_of_counts = [doc_freq_of_freqs[k] for k in doc_k]

    _, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

    # Token frequency distribution
    axes[0].bar(token_k, token_counts_of_counts)
    axes[0].set_title("Token Frequency Distribution")
    axes[0].set_xlabel("Total token frequency (k)")
    axes[0].set_ylabel("# tokens with frequency k")
    if log_y:
        axes[0].set_yscale("log")

    # Document frequency distribution
    axes[1].bar(doc_k, doc_counts_of_counts)
    axes[1].set_title("Token Document Frequency Distribution")
    axes[1].set_xlabel("# documents containing token (k)")
    axes[1].set_ylabel("# tokens with document frequency k")
    if log_y:
        axes[1].set_yscale("log")

    plt.tight_layout()
    plt.show()
