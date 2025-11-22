-- =============================================================================
-- lda_vocabulary.sql
--
-- Purpose
--   Store the global vocabulary derived from a cleaned news corpus for use in
--   LDA and related topic models, including per-term frequency statistics and
--   activation flags.
--
-- Row semantics
--   One row = one token (term) under a specific corpus_version, with its
--   global term count, document frequency, and inclusion status for training.
--
-- Conventions
--   - token is a normalized lexical unit produced by the corpus cleaning and
--     tokenization pipeline (e.g., case-folded, punctuation-stripped).
--   - corpus_version links this term definition to a particular corpus
--     construction run, consistent with lda_documents.corpus_version.
--   - global_term_count counts all appearances of the token across all
--     documents in the corpus_version.
--   - document_frequency counts how many distinct documents in the corpus
--     contain at least one occurrence of the token.
--
-- Keys & constraints
--   - Primary key: term_id (synthetic surrogate key).
--   - Uniqueness: (token, corpus_version) is unique to avoid duplicate term
--     definitions within the same corpus.
--   - Checks: token must be non-empty after TRIM; global_term_count ≥ 1;
--     document_frequency ≥ 1; global_term_count ≥ document_frequency.
--
-- Relationships
--   - Referenced by lda_document_terms.term_id for document–term counts.
--   - Implicitly aligned with lda_documents via shared corpus_version when
--     constructing bag-of-words representations.
--
-- Audit & provenance
--   - created_at records when the term row was inserted for a given corpus.
--   - Full details of the cleaning/tokenization process live in upstream
--     code and notebooks, not in this table.
--
-- Performance
--   - Index on (corpus_version, is_active) supports fast selection of the
--     active vocabulary for a particular corpus_version during LDA training.
--
-- Change management
--   - New corpus constructions should use new corpus_version values rather
--     than mutating existing rows, allowing historical vocabularies to
--     remain reproducible.
--   - Deactivating terms for modeling (e.g., boilerplate, rare terms) should
--     be done by toggling is_active, not by deleting rows.
-- =============================================================================
CREATE TABLE IF NOT EXISTS lda_vocabulary (

    -- ===========
    -- Identifiers
    -- ===========

    -- Token ID
    term_id SERIAL PRIMARY KEY,

    -- The token itself
    token TEXT NOT NULL,

    -- Corpus version
    corpus_version SMALLINT NOT NULL,

    -- ===================
    -- Token specific data
    -- ===================

    -- Global count of the token in the corpus
    global_term_count BIGINT NOT NULL,

    -- Document frequency of the token in the corpus
    document_frequency INTEGER NOT NULL,

    -- Is the token included in LDA training
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- ==========
    -- Provenance
    -- ==========
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- ===========
    -- Constraints
    -- ===========

    -- Trimmed token is not empty
    CONSTRAINT lda_vocab_chk_token_not_empty CHECK (LENGTH(TRIM(token)) > 0),

    -- Global term count is at least 1
    CONSTRAINT lda_vocab_chk_global_term_count CHECK (global_term_count >= 1),

    -- Document frequency is at least 1
    CONSTRAINT lda_vocab_chk_document_frequency CHECK (document_frequency >= 1),

    -- Global term count is at least document frequency
    CONSTRAINT lda_vocab_chk_term_count_vs_doc_freq CHECK
    (global_term_count >= document_frequency),

    -- Token is unique per corpus version
    CONSTRAINT lda_vocab_uk_token_corpus_version UNIQUE (token, corpus_version)
);

-- Index to speed up lookups by corpus version and active status
CREATE INDEX IF NOT EXISTS idx_lda_vocab_corpus_version_active
ON lda_vocabulary (corpus_version, is_active);

COMMENT ON TABLE lda_vocabulary IS
'Global vocabulary for a given corpus_version, including per-token frequency
statistics and an activation flag for LDA training.';

COMMENT ON COLUMN lda_vocabulary.term_id IS
'Synthetic surrogate key for the vocabulary term; used as a stable integer
identifier in document–term tables.';

COMMENT ON COLUMN lda_vocabulary.token IS
'Normalized lexical token produced by corpus cleaning and tokenization
(e.g., case-folded, punctuation-stripped); unique per corpus_version.';

COMMENT ON COLUMN lda_vocabulary.corpus_version IS
'Identifier of the corpus-construction run this term belongs to, matching
lda_documents.corpus_version.';

COMMENT ON COLUMN lda_vocabulary.global_term_count IS
'Total number of occurrences of this token across all documents in the
associated corpus_version.';

COMMENT ON COLUMN lda_vocabulary.document_frequency IS
'Number of distinct documents in the associated corpus_version that contain
at least one occurrence of this token.';

COMMENT ON COLUMN lda_vocabulary.is_active IS
'Boolean flag indicating whether this token is included in the active LDA
training vocabulary for the given corpus_version.';

COMMENT ON COLUMN lda_vocabulary.created_at IS
'UTC timestamp when this vocabulary term row was created for the given
corpus_version.';

COMMENT ON CONSTRAINT lda_vocab_uk_token_corpus_version ON lda_vocabulary IS
'Ensures that each (token, corpus_version) pair appears at most once in the
vocabulary.';

COMMENT ON INDEX idx_lda_vocab_corpus_version_active IS
'Index to accelerate retrieval of active tokens for a particular
corpus_version when constructing LDA training inputs.';
