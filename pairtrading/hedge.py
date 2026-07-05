"""
hedge.py  —  STEP 9: hedge ratio via linear regression
======================================================
Estimates the hedge ratio (beta) for each qualified pair by OLS:

        StockA_t = alpha + beta * StockB_t + e_t
        Spread_t = StockA_t - beta * StockB_t            (per spec formula)

We use statsmodels OLS.  This is mathematically identical to
sklearn.linear_model.LinearRegression (which does not yet ship a wheel for
Python 3.14); swapping it back in is a one-line change if desired.

We also compute the mean-reversion **half-life** (via an AR(1) fit on the
spread) — useful later for ranking pairs by "time to revert to the mean".

Output: data/hedge_ratio.csv  [Stock1, Stock2, Alpha, Beta, HalfLife]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import FILES
from .utils import log, save_csv


def ols_hedge(y: pd.Series, x: pd.Series) -> dict:
    """OLS of y on x (with intercept).  Returns alpha, beta and the spread."""
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    if len(df) < 20:
        return {"alpha": np.nan, "beta": np.nan, "spread": None}
    X = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], X).fit()
    alpha = float(model.params.get("const", np.nan))
    beta = float(model.params["x"])
    spread = df["y"] - beta * df["x"]          # spec: Spread = A - beta*B
    return {"alpha": alpha, "beta": beta, "spread": spread}


def half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion from an AR(1): d(spread) = k*spread_lag + c."""
    s = spread.dropna()
    if len(s) < 20:
        return np.nan
    lag = s.shift(1)
    delta = (s - lag).dropna()
    lag = lag.loc[delta.index]
    X = sm.add_constant(lag)
    try:
        k = sm.OLS(delta, X).fit().params.iloc[1]
        if k >= 0:                              # not mean-reverting
            return np.nan
        return float(-np.log(2) / k)
    except Exception:  # noqa: BLE001
        return np.nan


def compute_hedge_ratios(prices: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in pairs_df[["Stock1", "Stock2"]].itertuples(index=False):
        if a not in prices.columns or b not in prices.columns:
            continue
        res = ols_hedge(prices[a], prices[b])
        hl = half_life(res["spread"]) if res["spread"] is not None else np.nan
        rows.append(
            {"Stock1": a, "Stock2": b, "Alpha": res["alpha"],
             "Beta": res["beta"], "HalfLife": hl}
        )
    out = pd.DataFrame(rows)
    log.info("Hedge ratios computed for %d pairs.", len(out))
    save_csv(out, FILES["hedge"])
    return out
