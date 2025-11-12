-- =============================================================================
-- annual_fundamentals.sql
--
-- Purpose
--   Store firm-level accounting fundamentals extracted from 10-K/10-Q/20-F
--   filings at annual or quarterly frequency, keyed by CIK and fiscal period.
--   These values serve as raw inputs for constructing characteristics and
--   control variables (e.g., book-to-market, investment, profitability) in
--   downstream asset-pricing and news/returns research.
--
-- Row semantics
--   One row = one (CIK × fiscal_year × fiscal_period) observation as reported
--   in a specific SEC filing, with direct provenance back to the underlying
--   EDGAR evidence record.
--
-- Conventions
--   - CIK is a 10-digit, zero-padded TEXT identifier matching security_master.
--   - Fiscal periods are encoded as 'FY', 'Q1', 'Q2', 'Q3', 'Q4'; fiscal_year
--     is the reporting fiscal year, not necessarily the calendar year of
--     period_ending.
--   - Monetary items are stored as NUMERIC in the filing currency (e.g. USD)
--     without enforcing a particular unit (thousands, millions); unit handling
--     is left to the parser and downstream code.
--   - One EDGAR accession (and evidence_id) contributes at most one row to
--     this table, representing that filing's view of fundamentals for a given
--     CIK and period.
--
-- Keys & constraints
--   - Primary key: (cik, fiscal_year, fiscal_period).
--   - Natural keys / uniqueness:
--       * accession_num is unique per filing used here.
--       * evidence_id is unique per evidence row and can be used as a stable
--         provenance key.
--   - Checks:
--       * cik must match '^[0-9]{10}$' (numeric, zero-padded to 10 digits).
--       * fiscal_period must be one of ('FY', 'Q1', 'Q2', 'Q3', 'Q4').
--       * source_from_type is restricted to 10-K/10-Q/20-F and selected
--         amendment/transition variants.
--       * currency must be an ISO 4217 code (3 uppercase letters).
--       * Selected balance-sheet and flow items are constrained to be
--         non-negative where economically appropriate.
--
-- Relationships
--   - cik → security_master.cik (canonical entity registry).
--   - accession_num → EDGAR filing accession (mirrors ticker_cik_evidence
--     accession numbers for cross-reference).
--   - evidence_id → ticker_cik_evidence.evidence_id (provenance key for the
--     specific filing hit that produced this row).
--   - Downstream tables (e.g., derived_controls, factor_characteristics,
--     regression panels) are expected to join on (cik, fiscal_year,
--     fiscal_period) for panel work, or on evidence_id for filing-level
--     diagnostics.
--
-- Audit & provenance
--   - accession_num and evidence_id retain a direct link back to the original
--     EDGAR filing and the full raw_record stored in ticker_cik_evidence.
--   - ingested_at records the UTC load timestamp for this fundamentals row.
--   - No raw EDGAR payloads are stored here; this table is a cleaned,
--     research-ready layer above the evidence tables.
--
-- Performance
--   - The primary key on (cik, fiscal_year, fiscal_period) supports firm × time
--     lookups typical of panel regressions.
--
-- Change management
--   - Additive evolution is preferred: new accounting fields should be added
--     as nullable columns with clear semantics and unit conventions.
--   - Existing constraints on identifiers, currencies, and fiscal-period
--     semantics should be preserved; relaxing them requires reviewing
--     downstream joins and quality checks.
--   - Backfilling or revising historical rows should remain reproducible from
--     dated SEC filings and the evidence layer.
-- =============================================================================

CREATE TABLE IF NOT EXISTS annual_fundamentals (

    -- ===========
    -- Identifiers
    -- ===========

    -- CIK as a foreign key referencing security_master
    cik TEXT NOT NULL REFERENCES security_master (cik),

    -- Fiscal period ending date
    period_ending DATE NOT NULL,

    -- Fiscal year
    fiscal_year INTEGER NOT NULL,

    -- Fiscal period type (e.g., 'FY' for full year, 'Q1' for first quarter)
    fiscal_period TEXT NOT NULL,

    -- Source from type (e.g., '10-K', '10-Q')
    source_from_type TEXT NOT NULL,

    -- ==========
    -- Provenance
    -- ==========

    -- Accession number of the filing
    accession_num TEXT NOT NULL UNIQUE,

    -- Evidence ID for tracking data provenance
    evidence_id TEXT NOT NULL REFERENCES
    ticker_cik_evidence (evidence_id) UNIQUE,

    -- ===================
    -- Fundamental Metrics
    -- ===================

    -- Total assets
    total_assets NUMERIC NOT NULL,

    -- Total liabilities
    total_liabilities NUMERIC NOT NULL,

    -- Common equity
    common_equity NUMERIC NOT NULL,

    -- Shares outstanding
    shares_outstanding INTEGER NOT NULL,

    -- Net income
    net_income NUMERIC NOT NULL,

    -- Capital expenditures
    capital_expenditures NUMERIC NOT NULL,

    -- Depreciation and amortization
    depreciation_amortization NUMERIC NOT NULL,

    -- R&D expenses
    research_development_expenses NUMERIC NOT NULL,

    -- Gross property plant and equipment
    gross_property_plant_equipment NUMERIC NOT NULL,

    -- Inventories
    inventories NUMERIC NOT NULL,

    -- Income before extraordinary items
    income_before_extraordinary_items NUMERIC NOT NULL,

    -- Total revenue
    total_revenue NUMERIC NOT NULL,

    -- Cost of goods sold
    cost_of_goods_sold NUMERIC NOT NULL,

    -- Selling general and administrative expenses
    selling_general_administrative_expenses NUMERIC NOT NULL,

    -- Interest expenses
    interest_expenses NUMERIC NOT NULL,

    -- Total shareholder equity
    total_shareholder_equity NUMERIC NOT NULL,

    -- Preferred stock
    preferred_stock NUMERIC NOT NULL,

    -- Deferred taxes and investment tax credit
    deferred_taxes_investment_tax_credit NUMERIC NOT NULL,

    -- =========
    -- Meta Data
    -- =========

    -- Currency
    currency TEXT NOT NULL,

    -- Ingested at timestamp
    ingested_at TIMESTAMPTZ DEFAULT now(),

    -- ===========
    -- Constraints
    -- ===========

    -- Primary key constraint
    CONSTRAINT annual_fundamentals_pk
    PRIMARY KEY (cik, fiscal_year, fiscal_period),

    -- Ensure fiscal_period is 'FY' for annual filings
    CONSTRAINT annual_fundamentals_fiscal_period_yearly_check
    CHECK (
        (
            fiscal_period = 'FY'
            AND
            source_from_type IN ('10-K', '10-K/A', '10-K/T', '20-F', '20-F/A')
        )
        OR
        (
            fiscal_period IN ('Q1', 'Q2', 'Q3', 'Q4')
            AND source_from_type IN ('10-Q', '10-Q/A', '10-Q/T')
        )
    ),

    -- Ensure CIK format is numeric and 10 digits
    CONSTRAINT annual_fundamentals_cik_format CHECK (cik ~ '^[0-9]{10}$'),

    -- Ensure fiscal_period is one of the allowed values
    CONSTRAINT annual_fundamentals_fiscal_period_check
    CHECK (fiscal_period IN ('FY', 'Q1', 'Q2', 'Q3', 'Q4')),

    -- Ensure source type is one of the allowed values
    CONSTRAINT annual_fundamentals_source_type_check
    CHECK (
        source_from_type
        IN
        (
            '10-K', '10-K/A', '10-K/T', '10-Q',
            '10-Q/A', '10-Q/T', '20-F', '20-F/A'
        )
    ),

    -- Insure currency is ISO 4217 format (3 uppercase letters)
    CONSTRAINT annual_fundamentals_currency_format
    CHECK (currency ~ '^[A-Z]{3}$'),

    -- Check non-negative values for numeric fields
    CONSTRAINT annual_fundamentals_non_negative_check
    CHECK (
        (total_assets >= 0)
        AND
        (total_liabilities >= 0)
        AND
        (shares_outstanding >= 0)
        AND
        (capital_expenditures >= 0)
        AND
        (depreciation_amortization >= 0)
        AND
        (research_development_expenses >= 0)
        AND
        (gross_property_plant_equipment >= 0)
        AND
        (inventories >= 0)
        AND
        (total_revenue >= 0)
        AND
        (cost_of_goods_sold >= 0)
        AND
        (selling_general_administrative_expenses >= 0)
        AND
        (interest_expenses >= 0)
    )
);


COMMENT ON TABLE annual_fundamentals IS
'Firm-level accounting fundamentals by (CIK × fiscal_year × fiscal_period),
extracted from 10-K/10-Q/20-F filings and used as raw inputs for characteristics and controls.';

COMMENT ON COLUMN annual_fundamentals.cik IS
'SEC Central Index Key (CIK), 10-digit zero-padded TEXT;
entity anchor and FK to security_master.';

COMMENT ON COLUMN annual_fundamentals.period_ending IS
'Fiscal period ending date as reported in the filing
(calendar date of the statement).';

COMMENT ON COLUMN annual_fundamentals.fiscal_year IS
'Fiscal year label for the reporting period.';

COMMENT ON COLUMN annual_fundamentals.fiscal_period IS
'Fiscal period code: FY for full year, Q1–Q4 for interim quarters.';

COMMENT ON COLUMN annual_fundamentals.source_from_type IS
'Filing form type from which fundamentals were parsed
(e.g., 10-K, 10-Q, 20-F).';

COMMENT ON COLUMN annual_fundamentals.accession_num IS
'EDGAR accession number for the filing that supplied this fundamentals row;
unique per filing in this table.';

COMMENT ON COLUMN annual_fundamentals.evidence_id IS
'Stable provenance key referencing ticker_cik_evidence.evidence_id
for the filing hit that produced this row.';

COMMENT ON COLUMN annual_fundamentals.total_assets IS
'Total assets from the balance sheet in filing currency units;
stored as a non-negative NUMERIC.';

COMMENT ON COLUMN annual_fundamentals.total_liabilities IS
'Total liabilities from the balance sheet in filing currency units;
stored as a non-negative NUMERIC.';

COMMENT ON COLUMN annual_fundamentals.common_equity IS
'Common equity attributable to common shareholders (book value)
in filing currency units; may be negative for distressed firms.';

COMMENT ON COLUMN annual_fundamentals.shares_outstanding IS
'Shares of common stock outstanding for the reporting period;
non-negative INTEGER.';

COMMENT ON COLUMN annual_fundamentals.net_income IS
'Net income for the period in filing currency units,
including losses when negative.';

COMMENT ON COLUMN annual_fundamentals.capital_expenditures IS
'Capital expenditures for the period (e.g., additions to PPE)
in filing currency units, stored as a non-negative magnitude.';

COMMENT ON COLUMN annual_fundamentals.depreciation_amortization IS
'Depreciation and amortization expense for the period in filing currency units,
stored as a non-negative magnitude.';

COMMENT ON COLUMN annual_fundamentals.research_development_expenses IS
'Research and development (R&D) expense for the period in filing currency units;
non-negative NUMERIC when present.';

COMMENT ON COLUMN annual_fundamentals.gross_property_plant_equipment IS
'Gross property, plant, and equipment (before accumulated depreciation)
in filing currency units; non-negative NUMERIC.';

COMMENT ON COLUMN annual_fundamentals.inventories IS
'Inventory balance in filing currency units; non-negative NUMERIC.';

COMMENT ON COLUMN annual_fundamentals.income_before_extraordinary_items IS
'Income before extraordinary items
(or closest available pre-special-items earnings measure) in filing currency units; may be negative.';

COMMENT ON COLUMN annual_fundamentals.total_revenue IS
'Total revenue or net sales for the period in filing currency units;
stored as a non-negative magnitude.';

COMMENT ON COLUMN annual_fundamentals.cost_of_goods_sold IS
'Cost of goods sold (COGS) for the period in filing currency units;
non-negative magnitude used in gross profitability calculations.';

COMMENT ON COLUMN annual_fundamentals.selling_general_administrative_expenses IS
'Selling, general, and administrative (SG&A) expenses for the
period in filing currency units; non-negative magnitude.';

COMMENT ON COLUMN annual_fundamentals.interest_expenses IS
'Interest expense for the period in filing currency units;
non-negative magnitude used in profitability and leverage-related controls.';

COMMENT ON COLUMN annual_fundamentals.total_shareholder_equity IS
'Total shareholders equity from the balance sheet in filing currency units;
may be negative for firms with accumulated deficits.';

COMMENT ON COLUMN annual_fundamentals.preferred_stock IS
'Book value of preferred stock in filing currency units,
used in alternative book-equity definitions; may be zero or, in rare cases, negative.';

COMMENT ON COLUMN annual_fundamentals.deferred_taxes_investment_tax_credit IS
'Deferred taxes and investment tax credit component from the balance sheet
in filing currency units; can be positive or negative.';

COMMENT ON COLUMN annual_fundamentals.currency IS
'Filing currency code for all monetary fields on this row
(ISO 4217, e.g., USD, EUR).';

COMMENT ON COLUMN annual_fundamentals.ingested_at IS
'UTC load timestamp when this fundamentals row was
inserted into the research database.';
