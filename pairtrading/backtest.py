"""
backtest.py  —  STEP 11: vectorised event backtest per pair
===========================================================
Simulates the z-score mean-reversion strategy on every SELECTED pair and
produces both the individual trade log and per-pair performance statistics.

Return construction (dollar-neutral spread portfolio)
-----------------------------------------------------
A long-spread position holds +1 unit of A and -beta units of B.  Daily P&L is
the change in the spread, normalised by the gross notional deployed at the prior
close, so returns are expressed as a fraction of capital at risk:

    spread_change_t = dA_t - beta * dB_t
    gross_{t-1}     = A_{t-1} + |beta| * B_{t-1}
    ret_t           = pos_{t-1} * spread_change_t / gross_{t-1}
    net_ret_t       = ret_t - |dpos_t| * 2 * (txn_cost_bps / 1e4)   # 2 legs

`pos` is the causal signal LAGGED one bar (act next day => no look-ahead).

Metrics: cumulative return, annualised Sharpe, max drawdown, win %, number of
trades, and convergence-time stats (min/avg/max holding days).

Outputs
-------
data/trade_signals.csv    : one row per executed trade
data/backtest_results.csv : one row per pair (performance summary)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from config import CFG, FILES
from .returns import load_clean_matrix
from .signals import compute_spread, generate_positions, zscore
from .utils import log, save_csv

_PRICES: pd.DataFrame | None = None


def _init_worker(clean_csv_path: str) -> None:
    global _PRICES
    _PRICES = load_clean_matrix(clean_csv_path)


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan


def _simulate(a: str, b: str, beta: float) -> tuple[dict, list[dict]]:
    prices = _PRICES
    d = pd.concat([prices[a], prices[b]], axis=1).dropna()
    d.columns = ["A", "B"]
    if len(d) < CFG.zscore_window + 30 or not np.isfinite(beta):
        return ({"Stock1": a, "Stock2": b, "NumTrades": 0}, [])

    spread = compute_spread(d["A"], d["B"], beta)
    zf = zscore(spread, CFG.zscore_window)
    raw_pos = generate_positions(zf["Z"])
    pos = raw_pos.shift(1).fillna(0.0)                 # execute next bar

    dA = d["A"].diff()
    dB = d["B"].diff()
    spread_chg = dA - beta * dB
    gross = (d["A"].shift(1).abs() + abs(beta) * d["B"].shift(1).abs())
    gross = gross.replace(0, np.nan)

    ret = pos * spread_chg / gross
    dpos = pos.diff().abs().fillna(pos.abs())
    cost = dpos * 2.0 * (CFG.txn_cost_bps / 1e4)
    net = (ret - cost).fillna(0.0)

    equity = (1.0 + net).cumprod()

    # ---- per-trade decomposition (contiguous non-zero position runs) ------- #
    seg_id = (pos != pos.shift()).cumsum()
    trades: list[dict] = []
    for _, seg in pos.groupby(seg_id):
        side = seg.iloc[0]
        if side == 0:
            continue
        idx = seg.index
        seg_ret = float((1.0 + net.loc[idx]).prod() - 1.0)
        trades.append(
            {
                "Stock1": a, "Stock2": b,
                "Direction": "LONG_SPREAD" if side > 0 else "SHORT_SPREAD",
                "EntryDate": idx[0].date().isoformat(),
                "ExitDate": idx[-1].date().isoformat(),
                "HoldingDays": int(len(idx)),
                "EntryZ": float(zf["Z"].loc[idx[0]]),
                "ExitZ": float(zf["Z"].loc[idx[-1]]),
                "TradeReturnPct": round(seg_ret * 100, 4),
            }
        )

    # ---- pair-level statistics -------------------------------------------- #
    n_trades = len(trades)
    trade_rets = np.array([t["TradeReturnPct"] for t in trades], dtype=float)
    holding = np.array([t["HoldingDays"] for t in trades], dtype=float)
    ann_factor = np.sqrt(CFG.trading_days_per_year)
    vol = net.std()
    sharpe = float(net.mean() / vol * ann_factor) if vol and vol > 0 else np.nan

    stats = {
        "Stock1": a, "Stock2": b, "Beta": round(float(beta), 4),
        "TotalReturnPct": round(float(equity.iloc[-1] - 1.0) * 100, 4),
        "AnnReturnPct": round(float(net.mean() * CFG.trading_days_per_year) * 100, 4),
        "Sharpe": round(sharpe, 4) if np.isfinite(sharpe) else np.nan,
        "MaxDrawdownPct": round(_max_drawdown(equity) * 100, 4),
        "NumTrades": n_trades,
        "WinRatePct": round(float((trade_rets > 0).mean() * 100), 2) if n_trades else 0.0,
        "AvgTradeRetPct": round(float(trade_rets.mean()), 4) if n_trades else 0.0,
        "AvgHoldingDays": round(float(holding.mean()), 1) if n_trades else 0.0,
        "MinHoldingDays": int(holding.min()) if n_trades else 0,
        "MaxHoldingDays": int(holding.max()) if n_trades else 0,
    }
    return stats, trades


def _worker(task: tuple[str, str, float]) -> tuple[dict, list[dict]]:
    a, b, beta = task
    try:
        return _simulate(a, b, beta)
    except Exception as exc:  # noqa: BLE001
        return ({"Stock1": a, "Stock2": b, "NumTrades": 0, "error": str(exc)}, [])


def run_backtest(selected: pd.DataFrame) -> pd.DataFrame:
    tasks = [
        (r.Stock1, r.Stock2, float(r.Beta))
        for r in selected.itertuples(index=False)
        if np.isfinite(getattr(r, "Beta", np.nan))
    ]
    log.info("Backtesting %d selected pairs (workers=%d) ...", len(tasks), CFG.n_workers)

    if not tasks:
        save_csv(pd.DataFrame(columns=["Stock1", "Stock2"]), FILES["signals"])
        save_csv(pd.DataFrame(columns=["Stock1", "Stock2"]), FILES["backtest"])
        return pd.DataFrame()

    use_pool = CFG.n_workers > 1 and len(tasks) >= 20
    if use_pool:
        with ProcessPoolExecutor(
            max_workers=CFG.n_workers,
            initializer=_init_worker,
            initargs=(str(FILES["clean"]),),
        ) as ex:
            out = list(ex.map(_worker, tasks, chunksize=8))
    else:
        _init_worker(str(FILES["clean"]))
        out = [_worker(t) for t in tasks]

    stats = [s for s, _ in out]
    trades = [t for _, trs in out for t in trs]

    trades_df = pd.DataFrame(trades)
    save_csv(trades_df, FILES["signals"])

    stats_df = pd.DataFrame(stats).sort_values(
        "Sharpe", ascending=False, na_position="last"
    ).reset_index(drop=True)
    save_csv(stats_df, FILES["backtest"])
    log.info("Backtest complete: %d pairs, %d total trades.",
             len(stats_df), len(trades_df))
    return stats_df
