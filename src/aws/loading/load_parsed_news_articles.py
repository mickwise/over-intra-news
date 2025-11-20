import pandas as pd

from infra.utils.db_utils import connect_to_db, load_into_table


def load_parsed_news_articles(article_df: pd.DataFrame) -> None:
    query: str = generate_parsed_news_articles_query()
    generator = create_article_row_generator(article_df)
    with connect_to_db() as conn:
        load_into_table(conn, generator, query)


def generate_parsed_news_articles_query() -> str:
    return """
    INSERT INTO parsed_news_articles (
        article_id,
        trading_day,
        session,
        cik_list,
        warc_path,
        warc_date_utc,
        url,
        http_status,
        http_content_type,
        word_count,
        language_confidence,
        full_text
    ) VALUES %s ON CONFLICT (article_id) DO NOTHING;
    """


def create_article_row_generator(article_df: pd.DataFrame):
    for _, row in article_df.iterrows():
        yield (
            row["article_id"],
            row["ny_date"],
            row["session"],
            list(row["cik_list"]),
            row["warc_path"],
            row["warc_date_utc"],
            row["url"],
            row["http_status"],
            row["http_content_type"],
            row["word_count"],
            row["language_confidence"],
            row["full_text"],
        )
