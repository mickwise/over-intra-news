-- =============================================================================
-- lda_documents.sql
--
-- Purpose
--   Store cleaned, per-article document representations for LDA
--   versioned by corpus construction.
--
-- Row semantics
--   One row = one cleaned document view for a single article_id under a
--   specific corpus_version. The same article may appear in multiple rows
--   if it is reprocessed under different corpus definitions.
--
-- Conventions
--   - article_id matches parsed_news_articles.article_id.
--   - corpus_version is a small integer used to distinguish corpus
--     construction runs (e.g., different cleaning rules or vocabularies).
--
-- Keys & constraints
--   - Primary key: (article_id, corpus_version).
--   - Foreign key: article_id → parsed_news_articles(article_id).
--   - Checks: token_count ≥ 1; 1 ≤ unique_token_count ≤ token_count;
--
-- Relationships
--   - Referenced by lda_document_terms via (article_id, corpus_version)
--     to represent bag-of-words counts for each document.
--   - Joined to parsed_news_articles on article_id to recover trading_day,
--     session, and firm identifiers for downstream regressions.
--
-- Audit & provenance
--   - created_at records the ingestion time of this cleaned representation.
--   - Full parser and cleaning provenance lives upstream (e.g.,
--     parsed_news_articles and logging metadata), not in this table.
--
-- Performance
--   - Index on (corpus_version, included_in_training) supports efficient
--     selection of the active training subset for a given corpus_version.
--
-- Change management
--   - New corpus constructions should use new corpus_version values rather
--     than mutating existing rows, allowing multiple LDA runs to coexist.
--   - Existing rows are expected to be append-only aside from toggling
--     included_in_training when adjusting the training subset.
-- =============================================================================
CREATE TABLE IF NOT EXISTS lda_documents (

    -- ===========
    -- Identifiers
    -- ===========

    -- Article ID as found in parsed_news_articles.
    article_id TEXT REFERENCES parsed_news_articles (article_id),

    -- Corpus version
    corpus_version SMALLINT NOT NULL,

    -- Has the article been included in LDA training
    included_in_training BOOLEAN NOT NULL DEFAULT TRUE,

    -- =====================
    -- Article specific data
    -- =====================

    -- Number of tokens in the article after cleaning
    token_count INTEGER NOT NULL,

    -- Number of unique tokens in the article after cleaning
    unique_token_count INTEGER NOT NULL,

    -- ==========
    -- Provenance
    -- ==========
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- ===========
    -- Constraints
    -- ===========

    -- Primary key on article_id and corpus_version
    PRIMARY KEY (article_id, corpus_version),

    -- Token count must be at least 1
    CONSTRAINT lda_docs_chk_token_count CHECK (token_count >= 1),

    -- Unique token count must be at least 1 and not exceed token count
    CONSTRAINT lda_docs_chk_unique_token_count CHECK
    (unique_token_count >= 1 AND unique_token_count <= token_count)
);

-- Index on corpus_version, included_in_training for faster queries
CREATE INDEX IF NOT EXISTS idx_lda_documents_corpus_version_included
ON lda_documents (corpus_version, included_in_training);

COMMENT ON TABLE lda_documents IS
'Versioned, cleaned per-article documents used as the input corpus for LDA
and related topic-modeling experiments.';

COMMENT ON COLUMN lda_documents.article_id IS
'Identifier of the upstream article in parsed_news_articles; ties this
cleaned document back to its raw news record and metadata.';

COMMENT ON COLUMN lda_documents.corpus_version IS
'Small integer identifying the corpus-construction run (cleaning rules,
filters, vocabulary) under which this document representation was created.';

COMMENT ON COLUMN lda_documents.included_in_training IS
'Flag indicating whether this document is part of the LDA training set
for the associated corpus_version.';

COMMENT ON COLUMN lda_documents.token_count IS
'Total number of tokens in cleaned text after all corpus-level cleaning
and normalization steps have been applied.';

COMMENT ON COLUMN lda_documents.unique_token_count IS
'Number of distinct tokens present in the cleaned text; constrained to be at
least 1 and no greater than token_count.';


COMMENT ON COLUMN lda_documents.created_at IS
'UTC timestamp recording when this cleaned document row was created and
loaded into the LDA corpus tables.';

COMMENT ON INDEX idx_lda_documents_corpus_version_included IS
'Index to accelerate selection of documents by corpus_version and
included_in_training when assembling LDA training corpora.';
