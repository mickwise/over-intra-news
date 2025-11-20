-- =============================================================================
-- parsed_news_articles.sql
--
-- Purpose
--   Store cleaned, firm-linked news articles parsed from CC-NEWS WARC samples.
--   Each row represents a single article that passed all parser gating
--   conditions (HTML, length, language, firm match) and is aligned to a
--   NYSE trading day and session for integration with returns.
--
-- Row semantics
--   One row = one deduplicated article payload associated with a specific
--   NYSE trading day and session. Articles are already filtered to:
--   - HTTP 200 HTML responses,
--   - at least 25 tokens of visible ASCII text,
--   - langdetect probability >= 0.99 for English,
--   - between 1 and 3 matched firms.
--
-- Conventions
--   - trading_day is the NYSE trading date (America/New_York) sourced from
--     trading_calendar.trading_day.
--   - session is either 'intraday' or 'overnight', matching the return
--     decomposition used in the strategy.
--   - cik_list contains between 1 and 3 firm identifiers for which
--     the article is deemed relevant; ordering is not significant.
--   - all articles are assumed to be in English due to the language filter.
--
-- Keys & constraints
--   - Primary key: article_id.
--   - Natural uniqueness:
--       - article_id is derived from a hash of (trading_day, session,
--         full_text) to ensure deduplication within each (day, session).
--   - Checks:
--       - session must be 'intraday' or 'overnight'.
--       - cik_list length must be between 1 and 3.
--       - http_status must equal 200.
--       - word_count must be >= 25..
--
-- Relationships
--   - trading_day is expected to join to trading_calendar.trading_day to
--     align articles with session open/close times and returns.
--   - cik_list is expected to correspond to firm identifiers or CIKs
--     that can be joined to a security master or membership history table
--     (e.g., snp_membership, security_profile_history).
--   - Downstream topic-exposure and factor tables will typically join on
--     (trading_day, session, cik_list[*]) to aggregate articles into
--     firm × day × session features.
--
-- Audit & provenance
--   - warc_path, warc_date_utc, and url provide enough
--     provenance to trace each article back to its originating WARC sample
--     and HTTP response.
--   - Full parsing logs and run metadata live in the news parser run
--     registry and sample-level stats tables, not in this fact table.
--
-- Performance
--   - GIN index on cik_list supports efficient filtering by firm.
--   - The table is expected to be append-only over time with occasional
--     bulk reads for research and production scoring.
--
-- Change management
--   - Schema evolution should prefer add-only changes (new nullable columns)
--     to avoid breaking downstream consumers.
--   - Tightening constraints should be done with care and only after
--     validating existing data.
--   - If payload de-duplication rules change, consider adding new columns
--     or views rather than mutating historical rows.
-- =============================================================================

CREATE TABLE IF NOT EXISTS parsed_news_articles (
    -- ===========
    -- Identifiers
    -- ===========

    -- Unique identifier for each article
    article_id TEXT PRIMARY KEY,

    -- NYC trading day date
    trading_day DATE NOT NULL,

    -- Session (e.g. 'intraday', 'overnight')
    session TEXT NOT NULL,

    -- CIK list of firms mentioned in the article
    cik_list TEXT [] NOT NULL,

    -- ==========
    -- Provenance
    -- ==========

    -- WARC path
    warc_path TEXT NOT NULL,

    -- WARC date in UTC
    warc_date_utc TIMESTAMPTZ NOT NULL,

    -- URL from WARC-Target-URI header
    url TEXT NOT NULL,

    -- =============
    -- HTTP Metadata
    -- =============

    -- HTTP status code
    http_status INT NOT NULL,

    -- HTTP content type
    http_content_type TEXT NOT NULL,

    -- =========================
    -- Article Data and Metadata
    -- =========================

    -- Word count of the article
    word_count INT NOT NULL,

    -- Langdetect probability for the detected language
    language_confidence FLOAT NOT NULL,

    -- Article full text
    full_text TEXT NOT NULL,

    -- ===========
    -- Constraints
    -- ===========

    -- Ensure session is either 'intraday' or 'overnight'
    CONSTRAINT pna_chk_session CHECK (session IN ('intraday', 'overnight')),

    -- Ensure cik_list array has length between 1 and 3 (inclusive)
    CONSTRAINT pna_chk_cik_list_len CHECK
    (array_length(cik_list, 1) BETWEEN 1 AND 3),

    -- Ensure HTTP status code is 200
    CONSTRAINT pna_chk_http_status CHECK (http_status = 200),

    -- Ensure word count is greater or equal to 25
    CONSTRAINT pna_chk_word_count CHECK (word_count >= 25),

    -- Ensure language_confidence >= 0.99
    CONSTRAINT pna_chk_lang_conf_ge_99 CHECK
    (language_confidence >= 0.99)
);

-- Index on cik_list for faster querying
CREATE INDEX IF NOT EXISTS idx_cik_list
ON parsed_news_articles USING gin (cik_list);

COMMENT ON TABLE parsed_news_articles IS
'Cleaned, firm-linked news articles parsed from CC-NEWS WARC samples,
aligned to NYSE trading day and intraday/overnight sessions.';

COMMENT ON COLUMN parsed_news_articles.article_id IS
'Unique identifier for each article row.';

COMMENT ON COLUMN parsed_news_articles.trading_day IS
'NYSE trading date (America/New_York date) associated with the article,
used to align with returns and trading_calendar.trading_day.';

COMMENT ON COLUMN parsed_news_articles.session IS
'Session bucket for the article: either ''intraday'' or ''overnight'',
matching the return decomposition used downstream.';

COMMENT ON COLUMN parsed_news_articles.cik_list IS
'Array of firm identifiers (e.g., CIKs or surrogate firm_ids) matched to
this article; between 1 and 3 elements inclusive.';

COMMENT ON COLUMN parsed_news_articles.warc_path IS
'Full path/URI of the originating WARC file in storage (e.g., S3 key),
used for provenance and debugging; may appear in multiple rows.';

COMMENT ON COLUMN parsed_news_articles.warc_date_utc IS
'WARC-Date header for the response record, stored as a UTC timestamp
indicating when the payload was captured.';

COMMENT ON COLUMN parsed_news_articles.url IS
'Target URL of the article as recorded in the WARC-Target-URI header.';

COMMENT ON COLUMN parsed_news_articles.http_status IS
'HTTP status code of the response; constrained to 200 for kept articles.';

COMMENT ON COLUMN parsed_news_articles.http_content_type IS
'HTTP Content-Type header of the response (e.g., ''text/html; charset=utf-8'');
expected to contain an HTML media type for kept rows.';

COMMENT ON COLUMN parsed_news_articles.word_count IS
'Token count of the visible ASCII article text after parsing and cleaning;
must be at least 25.';

COMMENT ON COLUMN parsed_news_articles.language_confidence IS
'Langdetect probability for the detected language (expected to be English)';

COMMENT ON COLUMN parsed_news_articles.full_text IS
'Full cleaned article text stored as upper-case ASCII with non-visible
tags removed; used as the upstream corpus for topic modeling and NLP.';

COMMENT ON INDEX idx_cik_list IS
'GIN index on cik_list to speed queries filtering by firm identifier
(e.g., WHERE ''0000320193'' = ANY(cik_list)).';
