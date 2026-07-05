"""
cointegration.py  —  STEP 7 (Engle-Granger) + STEP 8 (ADF)
==========================================================
For every correlation-surviving pair we run:

STEP 7 - Engle-Granger cointegration on the PRICE LEVELS over three trailing
         windows: 1 year (252d), 120d and 60d.  A pair is "cointegrated" only
         if p-value < `coint_pvalue` in **all three** windows.

STEP 8 - Augmented Dickey-Fuller test on the pair's spread (StockA - beta*StockB,
         beta from full-sample OLS).  Stationary spread requires p-value < 0.05.

Both tests are genuinely per-pair and moderately expensive, so this stage is
parallelised with a process pool.  Each worker loads the clean price matrix once
(via an initializer) to avoid pickling it for every task.

Outputs
-------
data/cointegrated_pairs.csv : [Stock1, Stock2, Cointegration_1yr,
                               Cointegration_120d, Cointegration_60d] (all True)
data/adf_results.csv        : [Stock1, Stock2, ADF_statistic, ADF_pvalue]
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from config import CFG, FILES
from .returns import load_clean_matrix
from .utils import log, save_csv

# module-level handle populated inside each worker process
_PRICES: pd.DataFrame | None = None


def _init_worker(clean_csv_path: str) -> None:
    global _PRICES
    _PRICES = load_clean_matrix(clean_csv_path)


def _coint_pvalue(y: pd.Series, x: pd.Series) -> float:
    """Engle-Granger p-value; NaN if the test cannot be computed."""
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 30 or d.iloc[:, 0].std() == 0 or d.iloc[:, 1].std() == 0:
        return np.nan
    try:
        _, pval, _ = coint(d.iloc[:, 0], d.iloc[:, 1], trend="c", autolag="aic")
        return float(pval)
    except Exception:  # noqa: BLE001
        return np.nan


def _adf_on_spread(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Full-sample OLS beta -> spread -> ADF.  Returns (stat, pvalue)."""
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 30:
        return (np.nan, np.nan)
    yv, xv = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    beta = np.polyfit(xv, yv, 1)[0]           # slope only (matches OLS beta)
    spread = yv - beta * xv
    if np.std(spread) == 0:
        return (np.nan, np.nan)
    try:
        res = adfuller(spread, autolag="AIC")
        return (float(res[0]), float(res[1]))
    except Exception:  # noqa: BLE001
        return (np.nan, np.nan)


def _worker(pair: tuple[str, str]) -> dict:
    a, b = pair
    prices = _PRICES
    if prices is None or a not in prices.columns or b not in prices.columns:
        return {"Stock1": a, "Stock2": b}
    ya, xb = prices[a], prices[b]

    result: dict = {"Stock1": a, "Stock2": b}
    for label, win in CFG.coint_windows.items():
        yv, xv = ya.tail(win), xb.tail(win)
        result[f"pval_{label}"] = _coint_pvalue(yv, xv)

    stat, pval = _adf_on_spread(ya, xb)
    result["ADF_statistic"] = stat
    result["ADF_pvalue"] = pval
    return result


def run_cointegration(correlated_pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = list(correlated_pairs[["Stock1", "Stock2"]].itertuples(index=False, name=None))
    log.info("Cointegration+ADF on %d correlated pairs (workers=%d) ...",
             len(pairs), CFG.n_workers)

    if not pairs:
        empty = pd.DataFrame(columns=["Stock1", "Stock2"])
        save_csv(empty, FILES["cointegrated"])
        save_csv(empty, FILES["adf"])
        return pd.DataFrame()

    use_pool = CFG.n_workers > 1 and len(pairs) >= 50
    if use_pool:
        with ProcessPoolExecutor(
            max_workers=CFG.n_workers,
            initializer=_init_worker,
            initargs=(str(FILES["clean"]),),
        ) as ex:
            results = list(ex.map(_worker, pairs, chunksize=16))
    else:
        _init_worker(str(FILES["clean"]))
        results = [_worker(p) for p in pairs]

    res = pd.DataFrame(results)

    # ---- STEP 7 output: window flags + configurable gate ------------------- #
    for label in CFG.coint_windows:
        res[f"Cointegration_{label}"] = res[f"pval_{label}"] < CFG.coint_pvalue

    flag_cols = [f"Cointegration_{l}" for l in CFG.coint_windows]      # all 3 (diagnostics)
    gate_cols = [f"Cointegration_{l}" for l in CFG.coint_gate_windows]  # windows that MUST pass
    res["CointGatePass"] = res[gate_cols].all(axis=1)

    coint_out = res.loc[res["CointGatePass"], ["Stock1", "Stock2"] + flag_cols].reset_index(drop=True)
    save_csv(coint_out, FILES["cointegrated"])
    log.info("Cointegrated (gate=%s, diagnostics kept for all 3 windows): %d pairs.",
             "+".join(CFG.coint_gate_windows), len(coint_out))

    # ---- STEP 8 output: ADF for the gate-passing set ----------------------- #
    adf_out = res.loc[res["CointGatePass"], ["Stock1", "Stock2", "ADF_statistic", "ADF_pvalue"]].reset_index(drop=True)
    save_csv(adf_out, FILES["adf"])

    # attach ADF pass flag for downstream selection
    res["ADF_pass"] = res["ADF_pvalue"] < CFG.adf_pvalue
    return res
