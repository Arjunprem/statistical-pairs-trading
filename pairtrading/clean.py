"""
clean.py  —  STEP 3: build a clean, aligned price matrix
========================================================
Transforms the long historical master into a wide, analysis-ready matrix of
**Adjusted Close** prices (rows = trading dates, cols = symbols) and enforces
every cleaning rule from the spec.

Cleaning rules
--------------
1. Align all stocks on a common trading-date index (calendar = union of dates).
2. Drop duplicate dates / rows.
3. Invalidate non-positive prices (data errors) -> NaN.
4. Forward-fill only *small* gaps (<= ffill_limit consecutive days).
5. Drop stocks whose missing fraction exceeds `max_missing_pct`.
6. Drop stocks with fewer than `min_history_days` valid observations.
7. Drop dates that are still sparse after per-stock cleaning, so the final
   matrix has a consistent trading calendar with no residual gaps.

Output: data/clean_prices.csv  (Date index + one column per surviving symbol)
"""

from __future__ import annotations

import pandas as pd

from config import CFG, FILES
from .utils import log, save_csv


def _to_wide(long_df: pd.DataFrame, value_col: str = "Adj Close") -> pd.DataFrame:
    df = long_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    # rule 2: drop duplicate (Symbol, Date) keeping the last observation
    df = df.drop_duplicates(subset=["Symbol", "Date"], keep="last")
    wide = df.pivot(index="Date", columns="Symbol", values=value_col).sort_index()
    return wide


def clean_prices(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = _to_wide(long_df, "Adj Close")
    n0_dates, n0_syms = wide.shape
    log.info("Raw wide matrix: %d dates x %d symbols", n0_dates, n0_syms)

    # rule 3: non-positive prices are invalid -> NaN
    wide = wide.where(wide > 0)

    # rule 1: consistent calendar = the full date index already (union of dates)
    wide = wide[~wide.index.duplicated(keep="last")].sort_index()

    # rule 4: forward-fill only small gaps
    wide = wide.ffill(limit=CFG.ffill_limit)

    # rule 5: drop stocks with too many missing values
    missing_frac = wide.isna().mean()
    keep_missing = missing_frac[missing_frac <= CFG.max_missing_pct].index
    dropped_missing = sorted(set(wide.columns) - set(keep_missing))
    wide = wide[keep_missing]

    # rule 6: drop stocks with insufficient history
    valid_counts = wide.notna().sum()
    keep_hist = valid_counts[valid_counts >= CFG.min_history_days].index
    dropped_hist = sorted(set(wide.columns) - set(keep_hist))
    wide = wide[keep_hist]

    # rule 7: drop residually-sparse dates (need >=90% of stocks present),
    #         then final ffill of any tiny remaining interior gaps.
    row_coverage = wide.notna().mean(axis=1)
    wide = wide[row_coverage >= 0.90]
    wide = wide.ffill(limit=CFG.ffill_limit)
    # any stock still carrying NaNs after all of the above is dropped
    wide = wide.dropna(axis=1, how="any")

    log.info(
        "Cleaning dropped %d (missing) + %d (short history) symbols.",
        len(dropped_missing), len(dropped_hist),
    )
    if dropped_missing:
        log.info("  dropped for missingness: %s", ", ".join(dropped_missing[:20])
                 + (" ..." if len(dropped_missing) > 20 else ""))
    if dropped_hist:
        log.info("  dropped for short history: %s", ", ".join(dropped_hist[:20])
                 + (" ..." if len(dropped_hist) > 20 else ""))

    wide.index.name = "Date"
    log.info("Clean matrix: %d dates x %d symbols", *wide.shape)
    save_csv(wide.reset_index(), FILES["clean"], index=False)
    return wide


if __name__ == "__main__":
    long_df = pd.read_csv(FILES["historical"])
    clean_prices(long_df)
