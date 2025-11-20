from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from aws.loading.load_parsed_news_articles import load_parsed_news_articles
from infra.utils.db_utils import connect_to_db

BASE_ARTICLES = Path("local_data/ccnews_articles")
BASE_STATS = Path("local_data/ccnews_sample_stats")


def main():
    load_dotenv()
    load_all_articles()
    load_all_sample_stats()


def load_all_articles():
    for year_dir in sorted(BASE_ARTICLES.glob("year=*")):
        for month_dir in sorted(year_dir.glob("month=*")):
            for day_dir in sorted(month_dir.glob("day=*")):
                for session in ("intraday", "overnight"):
                    session_dir = day_dir / f"session={session}"
                    parquet_path = session_dir / "articles.parquet"
                    if not parquet_path.exists():
                        continue
                    articles_df: pd.DataFrame = pd.read_parquet(parquet_path)
                    articles_df = articles_df[articles_df['word_count'] >= 25]
                    articles_df["ny_date"] = pd.to_datetime(articles_df["ny_date"]).dt.date
                    load_parsed_news_articles(articles_df)


def load_all_sample_stats():
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            for year_dir in sorted(BASE_STATS.glob("year=*")):
                for month_dir in sorted(year_dir.glob("month=*")):
                    for day_dir in sorted(month_dir.glob("day=*")):
                        for session in ("intraday", "overnight"):
                            session_dir = day_dir / f"session={session}"
                            parquet_path = session_dir / "sample_stats.parquet"
                            if not parquet_path.exists():
                                continue
                            sample_stats_df: pd.DataFrame = pd.read_parquet(parquet_path)
                            trading_day = pd.to_datetime(sample_stats_df["date"].iloc[0]).date()
                            aggragated_data: tuple = (
                                trading_day,
                                session,
                                int(sample_stats_df["records_scanned"].sum()),
                                int(sample_stats_df["html_200_count"].sum()),
                                int(sample_stats_df["unhandled_errors"].sum()),
                                int(sample_stats_df["decompression_errors"].sum()),
                                int(sample_stats_df["ge_25_words"].sum()),
                                int(sample_stats_df["too_long_articles"].sum()),
                                int(sample_stats_df["english_count"].sum()),
                                int(sample_stats_df["matched_any_firm"].sum()),
                                int(sample_stats_df["articles_kept"].sum()),
                            )
                            query: str = generate_news_sample_stats_query()
                            cursor.execute(query, aggragated_data)
    conn.commit()



def generate_news_sample_stats_query() -> str:
    return """
    INSERT INTO news_sample_stats (
        trading_day,
        session,
        records_scanned,
        html_200_count,
        unhandled_errors,
        decompression_errors,
        ge_25_words,
        too_long_articles,
        english_count,
        matched_any_firm,
        articles_kept
    ) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (trading_day, session) DO NOTHING;
    """


if __name__ == "__main__":
    main()
