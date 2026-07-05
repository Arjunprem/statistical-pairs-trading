"""
returns.py  —  STEP 5: daily returns
=====================================
Simple daily returns  R_t = (P_t - P_{t-1}) / P_{t-1}  from the CLEAN adjusted
prices.  Correlation and every statistical filter downstream operate on these
*returns* (stationary-ish), never on raw price levels — computing correlation
on raw prices produces spurious results (both series trend), which the spec
explicitly forbids.
"""

from __future__ import annotations

import pandas as pd


def compute_returns(clean_wide: pd.DataFrame) -> pd.DataFrame:
    """clean_wide: Date-indexed matrix of adjusted close. Returns pct-change df."""
    if "Date" in clean_wide.columns:
        clean_wide = clean_wide.set_index("Date")
    rets = clean_wide.pct_change().iloc[1:]          # drop first NaN row
    return rets


def load_clean_matrix(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()
