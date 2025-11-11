# Public-Data Entity Resolution for S&P 500 Firms

## 1. Why build an S&P 500 entity layer (without CRSP or Bloomberg)

There’s a pretty common belief in quantitative finance that you can’t do serious work without vendor data: CRSP for returns and index membership, Bloomberg or a security master for entities, and so on. In particular, **entity resolution**—figuring out which *actual firm* a ticker refers to at a point in time—is usually delegated to those vendors.

The underlying research question is inspired by Glasserman, Krstovski, Laliberte and Mamaysky’s *“Does Overnight News Explain Overnight Returns?”* They decompose returns into intraday vs overnight components and relate each to the flow of news. To even start that analysis, you need a time-varying S&P 500 firm panel:

- For each **trading day**, you need to know **which firms** are in the index.
- For each firm, you need a **stable identity** so that “firm *i* at time *t*” means a specific legal entity, not “whatever happens to have ticker XYZ today.”
- For news, you also need consistent **company names** over time to drive name-based matching and sanity checks.

In this case, since data from such vendors was not available, these sources were used instead:

- **Ticker-level S&P 500 snapshots** from a public GitHub repository (`fja05680/sp500`).
- SEC **EDGAR** filings.
- **Wayback Machine** snapshots of the “List of S&P 500 companies” Wikipedia page to seed candidate CIKs for each S&P ticker episode.

Tickers, however, are mutable. A ticker can be reused by different firms over time, and a firm can change ticker without changing its underlying identity. A more stable ID was required.

On the SEC side, that object is the **CIK** (Central Index Key): a zero-padded, 10-digit identifier used across filings for a given legal entity. It isn’t perfect, but as a working assumption it is a much more stable firm key than ticker. The core problem then becomes:

> For each `(ticker, date)` in the S&P 500, assign a **CIK** and a **company name** as of that date, using only public data, in a way that is suitable for serious inference.

Everything else—topic models, predictive regressions, alpha—is layered on top of that.

---

## 2. From tickers to CIKs and names

Conceptually, the pipeline has three main inputs.

### 2.1 Ticker-level S&P 500 membership

The `fja05680/sp500` GitHub repo was used to build **ticker-based S&P 500 windows**: for each ticker, a half-open date range `[start, end)` over which it belongs to the index.

### 2.2 Wayback-derived candidate CIKs

For each `(ticker, validity_window)` derived from that GitHub data, Wayback snapshots of the Wikipedia page “List of S&P 500 companies” that fall inside the window were scraped, all CIKs that appear for that ticker were collected and stored as **candidate CIKs**.

This yields a set of possible entities for each S&P 500 ticker episode.

### 2.3 High-signal EDGAR evidence

For each `(ticker, validity_window, candidate_cik)` triple, a **restricted set of EDGAR forms** was pulled into `ticker_cik_evidence`: periodic reports and a curated slice of 8-K items that genuinely speak to issuer identity.

- Periodic reports (10-K/10-Q/20-F and variants) form the backbone: they can only be filed by the registrant and repeatedly restate name, legal entity, and exchange.
- Selected 8-K items (e.g. 2.01, 3.03, 5.01, 5.03) capture structural events like mergers, changes in control, or ticker/name changes inside a window.

The adjudication notebook formalizes this as an **evidence table** where each row is a filing supporting (or contradicting) that a given CIK is the actual issuer for an S&P 500 ticker episode.

From there:

- **Auto-accept rules** based on whether a candidate has in-window periodic filings were defined. For example, if a `(ticker, validity_window)` has only one candidate CIK and that candidate has at least one 10-K/10-Q/20-F within the window, that candidate is accepted without manual review.
- Ambiguous cases—multiple CIKs with periodic evidence, stray insider CIKs, noisy Wayback entries—are surfaced into a dedicated `ticker_cik_manual_adjudication` table, with a small set of actions (`seed_with_cik`, `manual_override`, `window_split`, `alias_rewrite`) and an explicit rationale plus URLs back to EDGAR and Wayback.

This yields a curated `ticker_cik_mapping` table with **exactly one primary CIK per `(ticker, validity_window)`** in the S&P 500 universe.

But CIK is only half the story. For news, **names** are also necessary.

The second notebook builds three CIK-keyed layers:

1. `security_master`: one row per CIK in the research universe (the entity registry).
2. `security_profile_history`: episodic name history per CIK, each row keyed by `(cik, validity_window)` with a canonical company name and non-overlap constraints.
3. `snp_membership`: a daily S&P 500 panel keyed by `(cik, trading_day)` derived by joining ticker-level S&P snapshots with `ticker_cik_mapping` and the trading calendar.

This design explicitly separates:

- **Who** the entity is (CIK, in `security_master`),
- **What it was called, and when** (`security_profile_history`),
- **When it was in the S&P 500** (`snp_membership` on `(cik, trading_day)`).

That separation is essential for news collection: it is possible to query news feeds by **names that are valid for a given CIK at a given time**, not by today’s name or today’s ticker.

---

## 3. Biases this pipeline is designed to avoid

This isn’t just elaborate ETL. It’s targeted at a very specific set of biases that can kill a news/returns study if you ignore them.

### 3.1 Survivorship bias

**Naive setup:**

- Use the current S&P 500 constituents (or a recent list) as your universe for all history.
- Drop firms that left the index or were delisted.

**Result:** overstating performance and misrepresenting cross-sectional distributions because the failures are gone.

**What was done instead:**

- Build **time-varying S&P 500 membership** from historical snapshots + trading calendar, then map tickers to CIKs and finally to a daily `(cik, trading_day)` panel in `snp_membership`.
- If a firm enters or leaves the index, that’s reflected as start/end dates in its membership series, not as a silent disappearance.

### 3.2 Ticker drift and ticker reuse

**Naive setup:**

- Treat ticker as the firm ID forever.

**Result:** the “same” ticker may correspond to completely different entities over time, especially across mergers, spin-offs, or reassignments.

**What was done instead:**

- Treat the S&P 500 inputs as **ticker episodes** with explicit `[start, end)` windows.
- Within each episode, resolve to a **CIK** via EDGAR evidence and adjudication.
- If the same ticker is recycled, it simply maps to a different CIK and window in `ticker_cik_mapping`.

### 3.3 Look-ahead bias in identity and membership

**Naive setup:**

- Use a modern vendor security master or index membership file compiled with full hindsight.
- Use today’s knowledge of which ticker/CIK pairs “belong together” even for past years.

**Result:** models implicitly know about future events when they shouldn’t.

**What was done instead:**

- Build the mapping from **documents that existed at the time**:
  - Historical S&P 500 snapshots from the GitHub repo.
  - Wayback snapshots that fall inside each S&P ticker window.
  - In-window EDGAR filings, with explicit `filed_at` timestamps.
- Enforce **as-of semantics** at the schema level:
  - `validity_window` columns use half-open ranges `[start, end)`.
  - For name history and membership, non-overlap constraints per CIK prevent contradictory episodes.

### 3.4 Name / identity confusion in news linking

**Naive setup:**

- Use whatever name your vendor gives you today to search a news archive.
- Hope that “IBM” means the same thing in 2001 and 2021.

**Result:** either miss relevant articles (because the name changed) or misattribute them.

**What was done instead:**

- Use `security_profile_history` to maintain a **canonical name episode per CIK over time**, with GiST constraints to enforce non-overlap.
- Apply a **company name canonicalizer** that strips punctuation and corporate/jurisdictional suffixes (“Inc.”, “PLC”, “LLC”, “DE”, etc.) so that superficial formatting changes don’t create fake name splits.
- Only the residual ~50–60 conflicting episodes are surfaced for manual review, e.g., genuine rebrands like “Facebook, Inc.” → “Meta Platforms, Inc.” or tricky strings where two different legal names are actually the same underlying issuer.

---

## 4. The pipeline, end-to-end

At a high level, the full entity-resolution pipeline looks like this:

1. **Ticker-level S&P 500 windows**
   - Use `fja05680/sp500` to build `[start, end)` validity windows per S&P 500 ticker.

2. **Wayback candidate generation**
   - For each `(ticker, validity_window)`, scrape Wayback snapshots of the “List of S&P 500 companies” page that fall in that window and collect all CIKs mentioned for that ticker.

3. **High-signal EDGAR evidence (`ticker_cik_evidence`)**
   - For each `(ticker, validity_window, candidate_cik)` triple, pull:
     - Periodic reports (10-K/10-Q/20-F and variants).
     - Selected 8-K items that encode structural changes (1.01, 2.01, 3.03, 5.01, 5.03).
   - Compute features like `has_periodic` per triple.

4. **Auto-accept rules & manual adjudication**
   - Apply deterministic rules:
     - Rule 1: single candidate with periodic filings → auto-accept.
     - Rule 2: multiple candidates, but exactly one with periodic filings → auto-accept that one.
   - Surface residuals (multi-candidate/multi-periodic episodes, stray CIKs, etc.) into `ticker_cik_manual_adjudication`, with actions like:
     - `seed_with_cik`
     - `manual_override`
     - `window_split`
     - `alias_rewrite`
     each with a rationale and links back to EDGAR/Wayback.

5. **Canonical mapping and evidence**
   - Collapse the results into `ticker_cik_mapping` with one primary CIK per `(ticker, validity_window)`.
   - For each resolved triple, pick a **canonical evidence filing** (e.g. prioritize 10-K/20-F, then more minor forms) to serve as the representative provenance record.

6. **Entity, names, membership (`security_master` notebook)**
   - Derive the CIK universe from `ticker_cik_mapping` and load it into `security_master`.
   - Build `security_profile_history` by:
     - Joining `ticker_cik_mapping` with `ticker_cik_evidence`.
     - Canonicalizing company names (`company_name_canonicalizer`).
     - Aggregating per (cik, canonical name) and deriving non-overlapping validity windows.
     - Auto-accepting single-name episodes; manually adjudicating the remaining conflicts.
   - Build `snp_membership` by combining:
     - Daily ticker-level S&P membership snapshots.
     - `ticker_cik_mapping` (to swap ticker for CIK).
     - The trading calendar (to ensure only valid trading days are included).

The result is a **CIK-keyed research layer**:

- One row per firm in `security_master`.
- Clean name episodes in `security_profile_history`.
- A daily S&P 500 membership panel in `snp_membership`.

Everything is built from public sources, with explicit provenance and a small, well-defined manual surface.

---

## 5. Summary visuals

### 5.1 Evidence review outcomes

Using the multi-candidate adjudication notebook, each `(ticker, validity_window)` episode is categorized by how it was resolved:

- **Rule 1** – single candidate CIK with in-window periodic filings.
- **Rule 2** – multiple candidates, but exactly one with in-window periodic filings.
- **Manual review** – anything that still can’t be resolved by periodic-based rules and lands in `ticker_cik_manual_adjudication`.

![Distribution of evidence review outcomes](notebooks/data_notebooks/Distribution_of_Auto_Accepted_Evidence_by_Rule.png)

The figure shows that the **vast majority of episodes are resolved deterministically** by periodic filings (Rule 1), a small additional slice by Rule 2, and only a **single-digit percentage** require manual adjudication. The manual layer is a narrow, explicitly audited tail rather than the main driver.

### 5.2 Name stability of candidate CIKs

In the security-master notebook, after applying `company_name_canonicalizer`, `name_count` is computed and the number of distinct canonical company names observed per `(ticker, validity_window, candidate_cik)`.

![Distribution of candidate CIKs by number of unique company names](notebooks/data_notebooks/Distribution_of_Company_Name_Counts.png)

The plot shows that **most candidate CIKs have a single canonical name** over their S&P 500 window. Only a small fraction exhibit multiple canonical names and need explicit inspection. In other words, once you strip suffixes and formatting, genuine name ambiguity is the exception, not the rule.

---

## 6. Why this matters for models

The original Glasserman et al. paper measures intraday vs overnight returns and links them to topic-modeled news, using CRSP for prices and constituents plus commercial security master data for identity.

My goal here is not to claim I’ve “improved” on their setup, but to show that **you can build a comparable entity layer from public data alone**, and that doing so is already a substantial piece of quant research:

- The **panel you estimate on** is different if you get entity resolution wrong.
- The **news exposures you compute** depend on whether you attach the right articles to the right firm at the right time.
- The **over-intra patterns you measure** (continuation, reversals, cross-sectional spreads) can be biased by survivorship, ticker drift, or look-ahead in the identity layer.

The meaning:

> If two people run the same “news explains overnight vs intraday returns” model, but only one of them has done this kind of entity work, they are **not** doing the same research.

This document is about that foundation:

- Taking S&P 500 tickers from a public GitHub repo,
- Using Wayback + EDGAR to resolve them to CIKs,
- Modeling names and membership as explicit time-varying layers,
- And enforcing as-of correctness and auditability *before* touching any return regression.