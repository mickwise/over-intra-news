import matplotlib.pyplot as plt
import pandas as pd


def summarize_dropped_firms(
    article_duration_df: pd.DataFrame,
    intensity_df: pd.DataFrame,
) -> None:
    real_news_days = (
        article_duration_df[article_duration_df["duration_days"].notnull()]
        .groupby("cik")["trading_day"]
        .nunique()
        .rename("n_article_days")
    )
    max_intensity_per_firm = intensity_df.groupby("cik")["yearly_intensity"].max()
    has_dynamic_model = max_intensity_per_firm > 0.0
    summary_df = pd.DataFrame(
        {"n_article_days": real_news_days, "has_dynamic_model": has_dynamic_model}
    )
    summary_df["n_article_days"] = summary_df["n_article_days"].fillna(0)
    summary_df["has_dynamic_model"] = summary_df["has_dynamic_model"].fillna(False)
    n_total = summary_df.shape[0]
    n_dynamic = int(summary_df["has_dynamic_model"].sum())
    n_static = n_total - n_dynamic

    print(f"Total firms in universe: {n_total}")
    print(f"Firms with Dynamic ACD Model (Tier 1/2): {n_dynamic}")
    print(f"Firms with Static Zero Intensity (Tier 3): {n_static}\n")
    print("Article-day counts by Model Type:\n")
    print(summary_df.groupby("has_dynamic_model")["n_article_days"].describe())

    dynamic_counts = summary_df.loc[summary_df["has_dynamic_model"], "n_article_days"]
    static_counts = summary_df.loc[~summary_df["has_dynamic_model"], "n_article_days"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    if not dynamic_counts.empty:
        axes[0].hist(dynamic_counts.values, bins=30, color="tab:blue", alpha=0.7)
    axes[0].set_title(f"Dynamic Intensity (N={n_dynamic})\n(Converged ACD Models)")
    axes[0].set_xlabel("Number of article days")
    axes[0].set_ylabel("Count of firms")

    if not static_counts.empty:
        axes[1].hist(static_counts.values, bins=30, color="tab:orange", alpha=0.7)
    axes[1].set_title(f"Static Zero Intensity (N={n_static})\n(Low Info / Phantom Firms)")
    axes[1].set_xlabel("Number of article days")

    fig.tight_layout()
    plt.show()
