"""
signals.py  —  STEP 10: spread, rolling z-score, trading signals
================================================================
For a pair (A, B) with hedge ratio beta:

        Spread_t     = A_t - beta * B_t
        RollingMean  = SMA(Spread, window)
        RollingStd   = STD(Spread, window)
        Z_t          = (Spread_t - RollingMean_t) / RollingStd_t

Trading logic (mean-reversion)
------------------------------
        Z >= +2  -> SHORT the spread  (short A, long beta*B)   position = -1
        Z <= -2  -> LONG  the spread  (long A, short beta*B)   position = +1
        |Z| back to ~0 -> EXIT (flat)
        |Z| >= z_stop  -> STOP OUT (risk guard for non-reverting blow-ups)

Positions are generated causally and then LAGGED one day in the backtest, so a
signal observed at close_t is acted on at close_{t+1} (no look-ahead bias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG


def compute_spread(a: pd.Series, b: pd.Series, beta: float) -> pd.Series:
    return a - beta * b


def zscore(spread: pd.Series, window: int = CFG.zscore_window) -> pd.DataFrame:
    roll_mean = spread.rolling(window).mean()
    roll_std = spread.rolling(window).std()
    z = (spread - roll_mean) / roll_std
    return pd.DataFrame(
        {"Spread": spread, "RollingMean": roll_mean, "RollingStd": roll_std, "Z": z}
    )


def generate_positions(z: pd.Series,
                       entry: float = CFG.z_entry,
                       exit_: float = CFG.z_exit,
                       stop: float = CFG.z_stop) -> pd.Series:
    """
    Causal state machine -> desired position (+1 long spread, -1 short, 0 flat)
    on the SAME bar the signal is observed.  The backtester lags this by one bar.
    """
    pos = np.zeros(len(z), dtype=float)
    state = 0
    zv = z.to_numpy()
    for i, zi in enumerate(zv):
        if np.isnan(zi):
            pos[i] = 0.0
            state = 0
            continue
        if state == 0:                       # flat -> look for entry
            if zi >= entry:
                state = -1                    # short the (rich) spread
            elif zi <= -entry:
                state = +1                    # long the (cheap) spread
        elif state == +1:                    # long spread -> exit on revert/stop
            if zi >= exit_ or zi <= -stop:
                state = 0
        elif state == -1:                    # short spread -> exit on revert/stop
            if zi <= exit_ or zi >= stop:
                state = 0
        pos[i] = state
    return pd.Series(pos, index=z.index, name="Position")
