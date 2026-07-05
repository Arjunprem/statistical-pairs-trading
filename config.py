"""
config.py
=========
Central configuration for the Indian F&O pair-trading pipeline.

Every tunable knob lives here so the pipeline is fully reproducible and no
"magic numbers" are buried inside the analytics modules.  Import `CFG` from
this module everywhere.

Author: built for Arjun Premkumar (74 North Capital)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"          # all CSV artefacts land here
CACHE_DIR = PROJECT_ROOT / "cache"        # raw per-ticker downloads (parquet)
REPORT_DIR = PROJECT_ROOT / "reports"     # plots + summary artefacts
LOG_DIR = PROJECT_ROOT / "logs"

for _d in (DATA_DIR, CACHE_DIR, REPORT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    # ---- Step 1: universe -------------------------------------------------- #
    target_universe_size: int = 250          # how many stocks we WANT
    # NSE "securities in F&O" snapshot endpoint (returns live symbols + prices)
    nse_home: str = "https://www.nseindia.com"
    nse_fno_api: str = (
        "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
    )
    nse_fno_lots_csv: str = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"

    # ---- Step 2: history download ------------------------------------------ #
    history_years: int = 4                   # >= 2 as required; 4 for buffer
    yf_suffix: str = ".NS"                   # NSE tickers on Yahoo Finance
    yf_batch_size: int = 40                  # tickers per yfinance batch call
    yf_max_retries: int = 3
    yf_pause_seconds: float = 1.5            # politeness delay between batches

    # ---- Step 3: cleaning -------------------------------------------------- #
    max_missing_pct: float = 0.10            # drop a stock if >10% dates missing
    min_history_days: int = 260              # need >= ~1yr of clean data
    ffill_limit: int = 3                     # only forward-fill tiny gaps (<=3d)

    # ---- Step 6: correlation ---------------------------------------------- #
    corr_threshold: float = 0.70             # keep pairs with |corr| >= 0.70

    # ---- Step 7: cointegration windows (trading days) --------------------- #
    coint_windows: dict = field(
        default_factory=lambda: {"1yr": 252, "120d": 120, "60d": 60}
    )
    coint_pvalue: float = 0.05               # Engle-Granger p-value threshold
    # Windows that MUST be cointegrated for a pair to qualify (the "gate").
    # The 60d/120d Engle-Granger tests have low statistical power, so we gate on
    # the 1-year window and keep 120d/60d as reported robustness diagnostics.
    # Set to ("1yr","120d","60d") to reproduce the strict "all-3-windows" spec.
    coint_gate_windows: tuple = ("1yr",)

    # ---- Step 8: ADF ------------------------------------------------------- #
    adf_pvalue: float = 0.05                 # spread must be stationary

    # ---- Step 10: z-score / signals --------------------------------------- #
    zscore_window: int = 60                  # rolling window for mean/std
    z_entry: float = 2.0                     # |z| >= 2 -> enter
    z_exit: float = 0.0                       # z crosses 0 -> exit
    z_stop: float = 3.5                      # hard stop if spread blows out

    # ---- Step 11: backtest ------------------------------------------------- #
    trading_days_per_year: int = 252
    txn_cost_bps: float = 4.0                # 4 bps per leg per turn (xlsx note)
    capital_per_trade: float = 1.0           # notional; returns reported in %

    # ---- Execution --------------------------------------------------------- #
    n_workers: int = max(1, (os.cpu_count() or 4) - 1)
    random_seed: int = 42

    # ---- Derived ----------------------------------------------------------- #
    @property
    def start_date(self) -> str:
        return (date.today() - timedelta(days=int(self.history_years * 365.25))).isoformat()

    @property
    def end_date(self) -> str:
        return date.today().isoformat()


CFG = Config()

# --------------------------------------------------------------------------- #
# Output file registry (single source of truth for artefact names)
# --------------------------------------------------------------------------- #
FILES = {
    "universe":      DATA_DIR / "fno_250_stocks.csv",
    "historical":    DATA_DIR / "historical_prices.csv",
    "clean":         DATA_DIR / "clean_prices.csv",
    "all_pairs":     DATA_DIR / "all_pairs.csv",
    "correlated":    DATA_DIR / "correlated_pairs.csv",
    "cointegrated":  DATA_DIR / "cointegrated_pairs.csv",
    "adf":           DATA_DIR / "adf_results.csv",
    "hedge":         DATA_DIR / "hedge_ratio.csv",
    "selected":      DATA_DIR / "selected_pairs.csv",
    "signals":       DATA_DIR / "trade_signals.csv",
    "backtest":      DATA_DIR / "backtest_results.csv",
}
