# Schema Plan

## Reference tables

### trading_calendar
- Row: One NYSE trading day
- Primary key: trading_day
- Why: Defines NYSE open/close for intraday vs overnight and rolls weekends/holidays to the next trading day.

### security_master
- Row: One S&P 500 company/security in your universe
- Primary key: secid
- Why: Stable internal ID that doesn’t change when tickers/mergers happen.

### ticker_history
- Row: One ticker for a company over a validity window
- Primary key: (secid, ticker, effective_start)
- Why: Maps secid to ticker across time so joins don’t break.

### sp500_membership
- Row: One company on one trading day
- Primary key: (secid, trading_day)
- Why: Daily membership to avoid survivorship bias and to define the investable universe.

### corporate_actions
- Row: One corporate action for a company on an effective date
- Primary key: (secid, ex_date, action_type)
- Why: Splits/dividends/buybacks to reconstruct open/close returns if vendor adjustments don’t match the paper’s conventions.

## Time-series tables

### prices_daily
- Row: One company on one trading day (OHLCV)
- Primary key: (secid, trading_day)
- Why: Source for intraday (open→close) and overnight (close→next open) returns; base for momentum/volatility.

### topic_exposure_daily
- Row: One company × trading day × period (i|o) × topic
- Primary key: (secid, trading_day, period, topic_id)
- Why: Sum of article-level topic probabilities for that firm within the session (intraday/overnight).

### topic_exposure_yearly
- Row: One company × model year t × period × topic
- Primary key: (secid, year_t, period, topic_id)
- Why: Yearly cumulative exposures used in cross-sectional regressions and forecasts.

### controls_yearly
- Row: One company × model year t
- Primary key: (secid, year_t)
- Why: Stores size, book-to-market (asinh if used), investment, profitability, 12-month momentum (skip 1), and intraday/overnight vol (raw + demeaned).

## Model/artifact tables

### news_article
- Row: One news article
- Primary key: article_id
- Why: Holds publish timestamp (UTC), source, headline, body text/raw JSON; used to assign the article to a trading day & period and link it to firms.

### article_company_link
- Row: One (article, company) mention
- Primary key: (article_id, secid)
- Why: Connects articles to firms (provider tags or NER); basis for aggregating exposures.

### lda_model
- Row: One frozen LDA
- Primary key: model_id
- Why: Records K (e.g., 200), training window, seed, preprocessing parameters; ensures reproducibility/versioning.

### lda_art_topic
- Row: One (article, topic) probability from a specific model vintage
- Primary key: (article_id, topic_id)
- Why: Document–topic distribution; atomic inputs summed into firm/session exposures.

## Relationships (how tables connect)

- Article → Firm:
  - `news_article (article_id)` → `article_company_link (article_id, secid)` → `security_master (secid)`
  - `identifier_history` resolves provider identifiers/tickers to `secid` using `effective_start/effective_end`.

- Article → Session (intraday vs overnight):
  - `news_article.published_ts_utc` + `trading_calendar` define `(trading_day, period)`.
  - Convert publish time to America/New_York; compare to that day’s `open`/`close`.
  - If weekend/holiday, assign to the **next** trading day’s **overnight**.

- Prices & Universe:
  - `prices_daily (secid, trading_day)` joins to `sp500_membership (secid, trading_day)` to restrict tests to in-universe firms for that day/year.
  - `corporate_actions` used only if vendor “adjusted” fields don’t match the paper’s conventions.

- Topics & Exposures:
  - `lda_art_topic (article_id, topic_id, prob)` aggregates via `article_company_link` into
    `topic_exposure_daily (secid, trading_day, period, topic_id)` by summing probs for all articles of that firm in that session.
  - Roll up to `topic_exposure_yearly (secid, year_t, period, topic_id)` via the paper’s annual window (cumulative exposures for year t).

- Controls:
  - `controls_yearly (secid, year_t)` computed from fundamentals + prices (momentum/vol), joined to exposures by `(secid, year_t)`.

- Model lineage:
  - `lda_art_topic` references a single `lda_model (model_id)`; a model “vintage” is immutable once used in forecasts.


## Invariants (data rules that must always hold)

1) **Unique session mapping:** every `news_article` used for modeling maps to **exactly one** `(trading_day, period ∈ {i,o})` per `trading_calendar` rules.

2) **Article filters:** articles with **<25 words** or mentioning **>3 S&P 500 firms** are **excluded** from exposure construction (keep a flag if you store them).

3) **Price uniqueness:** `prices_daily` has at most one row per `(secid, trading_day)`; when both `open` and `close` exist, you can compute `r_i` (open→close) and `r_o` (close→next open).

4) **Corporate-action convention:** dividends accrue to the **overnight** leg (close→open); splits/buybacks take effect **at the close**. Vendor adjustments must align; otherwise reconstruct using `corporate_actions`.

5) **Universe discipline:** `topic_exposure_daily` and portfolio tests use only firms in `sp500_membership` on that `trading_day` (state explicitly if you keep out-of-universe articles for context).

6) **Topic model freeze:** each `lda_model` (K, seed, training window, preprocessing) is **immutable** once referenced by `lda_art_topic` or any forecast.

7) **Controls treatment:** `controls_yearly` stores both **raw** values and **cross-sectionally demeaned** values within year t; the modeling uses the demeaned set.

8) **Time policy:** all timestamps stored as **UTC**; intraday/overnight boundaries derived from `trading_calendar` (America/New_York, DST-safe).
