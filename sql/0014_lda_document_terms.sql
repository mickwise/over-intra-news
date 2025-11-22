-- =============================================================================
-- lda_document_terms.sql
--
-- Purpose
--   Store the sparse document–term matrix
--   linking cleaned articles to vocabulary
--   terms for a given corpus_version,
--   with per-document term counts suitable for
--   LDA and related topic models.
--
-- Row semantics
--   One row = one (article_id, corpus_version, term_id) triple with the count
--   of how many times that term appears in that article under the specified
--   corpus_version.
--
-- Conventions
--   - article_id and corpus_version together identify a single cleaned document
--     in lda_documents.
--   - term_id references a single vocabulary entry in lda_vocabulary.
--   - term_count is the raw (non-normalized) count of the token in the
--     cleaned_text associated with (article_id, corpus_version).
--   - Rows are typically append-only within a corpus_version once constructed,
--     so that model inputs remain reproducible.
--
-- Keys & constraints
--   - Primary key: (article_id, corpus_version, term_id).
--   - Foreign keys:
--       * (article_id, corpus_version) →
--         lda_documents(article_id, corpus_version)
--       * term_id → lda_vocabulary(term_id)
--   - Checks: term_count ≥ 1 to exclude zero-count entries.
--
-- Relationships
--   - Bridges lda_documents and lda_vocabulary to form the bag-of-words
--     representation consumed by LDA.
--   - Joined with lda_documents to recover document-level metadata
--     (e.g., trading_day, session, CIKs) for downstream regressions.
--   - Joined with lda_vocabulary to recover token strings and frequency
--     statistics for diagnostics and vocabulary curation.
--
-- Audit & provenance
--   - Provenance for documents (cleaning pipeline, corpus construction) is
--     tracked in lda_documents and upstream notebooks; this table stores only
--     the resulting counts.
--   - Vocabulary-level provenance lives in lda_vocabulary.
--
-- Performance
--   - Primary key index supports efficient lookups of all term counts for a
--     given (article_id, corpus_version).
--   - Secondary index on (term_id, corpus_version) accelerates reverse
--     lookups of all documents containing a given term in a given corpus.
--
-- Change management
--   - New corpus constructions should use new corpus_version values rather than
--     mutating existing rows, preserving reproducibility of past experiments.
--   - Term pruning for modeling (e.g., dropping rare or boilerplate terms)
--     should be handled by toggling lda_vocabulary.is_active or by rebuilding
--     a new corpus_version, not by in-place deletion from this table.
-- =============================================================================
CREATE TABLE IF NOT EXISTS lda_document_terms (
    -- ===========
    -- Identifiers
    -- ===========

    -- Article ID and corpus version composite foreign key
    article_id TEXT,
    corpus_version SMALLINT,
    FOREIGN KEY (article_id, corpus_version)
    REFERENCES lda_documents (article_id, corpus_version) ON DELETE CASCADE,

    -- Token ID
    term_id INTEGER REFERENCES lda_vocabulary (term_id) ON DELETE CASCADE,

    -- =========================
    -- Article-Term specific data
    -- ==========================

    -- Count of the term in the article
    term_count INTEGER NOT NULL,

    -- ===========
    -- Constraints
    -- ===========

    -- Primary key on article_id, corpus_version, and term_id
    PRIMARY KEY (article_id, corpus_version, term_id),

    -- Term count must be at least 1
    CONSTRAINT lda_doc_terms_chk_term_count CHECK (term_count >= 1)
);

-- Index to optimize queries filtering by term_id and corpus_version
CREATE INDEX IF NOT EXISTS idx_lda_doc_terms_term_corpus
ON lda_document_terms (term_id, corpus_version);

COMMENT ON TABLE lda_document_terms IS
'Sparse document–term matrix linking cleaned articles to vocabulary terms for
a given corpus_version, with per-document term counts for LDA.';

COMMENT ON COLUMN lda_document_terms.article_id IS
'Article identifier, matching parsed_news_articles.article_id and paired
with corpus_version to reference a single cleaned document in lda_documents.';

COMMENT ON COLUMN lda_document_terms.corpus_version IS
'Corpus construction identifier, used together with article_id to select
the appropriate cleaned document representation.';

COMMENT ON COLUMN lda_document_terms.term_id IS
'Foreign key to lda_vocabulary.term_id identifying the token whose count
is recorded in this row.';

COMMENT ON COLUMN lda_document_terms.term_count IS
'Number of occurrences of the given term_id in the cleaned text of
(article_id, corpus_version); constrained to be at least 1.';

COMMENT ON CONSTRAINT lda_doc_terms_chk_term_count ON lda_document_terms IS
'Ensures that only strictly positive term counts are stored in the
document–term matrix.';

COMMENT ON INDEX idx_lda_doc_terms_term_corpus IS
'Index to speed up reverse lookups of all documents containing a given
term_id within a particular corpus_version.';
