"""
Purpose
-------
Provide shared configuration constants for time handling and calendar padding
used across the CC-NEWS sampling pipeline.

Key behaviors
-------------
- Fixes the canonical civil timezone for date bucketing (`DATE_TZ`).
- Defines the canonical civil-date string format (`DATE_FMT`).
- Exposes the number of days of padding (`PAD_DAYS`) applied around month
  boundaries when querying the trading calendar.
- Specifies the last civil date for which sampling is performed
  (`LAST_SAMPLING_DAY`).

Conventions
-----------
- `DATE_TZ` is the timezone used for civil trading dates; individual session
  instants are handled in UTC elsewhere.
- `DATE_FMT` is a `%Y-%m-%d` strftime/strptime pattern for New York civil dates.
- `PAD_DAYS` is a `datetime.timedelta` representing whole-day padding; any
  horizon-edge exceptions are implemented by callers (e.g., in `calendar_utils`).
- `LAST_SAMPLING_DAY` is the final civil date for which sampling is performed.

Downstream usage
----------------
Import these constants as read-only configuration:

    from aws.ccnews_sampler.ccnews_sampler_config import DATE_TZ, DATE_FMT, PAD_DAYS

Treat them as immutable at runtime; adjust them here to change behavior
consistently across the pipeline.
"""

import datetime as dt

DATE_TZ: str = "America/New_York"
DATE_FMT: str = "%Y-%m-%d"
PAD_DAYS: dt.timedelta = dt.timedelta(days=7)
LAST_TRADING_DAY_PRE_SAMPLE: str = "2016-07-29"
LAST_SAMPLING_DAY: dt.date = dt.date(2025, 8, 1)
