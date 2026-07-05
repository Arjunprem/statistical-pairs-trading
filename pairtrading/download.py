"""
download.py  —  STEP 2: download historical daily data from Yahoo Finance
=========================================================================
For every symbol in the universe we pull ~`history_years` of daily OHLCV via
yfinance (NSE tickers use the ``.NS`` suffix).  We keep BOTH raw ``Close`` and
``Adj Close`` (auto_adjust=False), as the spec requires.

Robustness features
-------------------
* per-ticker CSV cache  -> reruns are instant, partial progress survives crashes
* batched downloads      -> fewer HTTP round-trips
* per-ticker retry       -> a flaky batch does not kill good tickers
* every failure logged, never raised (a few dead tickers are expected)

Output: data/historical_prices.csv
        long format: [Symbol, Date, Open, High, Low, Close, Adj Close, Volume]
"""

from __future__ import annotations

import time
import warnings

import pandas as pd
import yfinance as yf

from config import CFG, CACHE_DIR, FILES
from .utils import log, save_csv, exists

warnings.simplefilter("ignore", category=FutureWarning)

_OHLCV = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _cache_path(symbol: str):
    return CACHE_DIR / f"{symbol}.csv"


def _extract_one(raw: pd.DataFrame, yf_ticker: str) -> pd.DataFrame | None:
    """Pull a single ticker's OHLCV frame out of a (possibly multi-index) download."""
    if raw is None or raw.empty:
        return None
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if yf_ticker not in raw.columns.get_level_values(0):
                return None
            sub = raw[yf_ticker].copy()
        else:
            sub = raw.copy()
    except Exception:  # noqa: BLE001
        return None

    sub = sub.dropna(how="all")
    if sub.empty:
        return None
    # ensure all expected columns exist (Adj Close missing if auto_adjust slipped)
    for col in _OHLCV:
        if col not in sub.columns:
            if col == "Adj Close" and "Close" in sub.columns:
                sub["Adj Close"] = sub["Close"]
            else:
                sub[col] = pd.NA
    sub = sub[_OHLCV]
    sub.index.name = "Date"
    return sub.reset_index()


def _download_batch(ns_tickers: list[str]) -> pd.DataFrame:
    return yf.download(
        tickers=ns_tickers,
        start=CFG.start_date,
        end=CFG.end_date,
        interval="1d",
        auto_adjust=False,      # keep raw Close AND Adj Close
        group_by="ticker",
        threads=True,
        progress=False,
    )


def download_prices(symbols: list[str], force: bool = False) -> pd.DataFrame:
    to_fetch = [s for s in symbols if force or not exists(_cache_path(s))]
    cached = len(symbols) - len(to_fetch)
    log.info("Download: %d symbols total | %d cached | %d to fetch",
             len(symbols), cached, len(to_fetch))

    failed: list[str] = []
    for i in range(0, len(to_fetch), CFG.yf_batch_size):
        batch = to_fetch[i:i + CFG.yf_batch_size]
        ns_map = {f"{s}{CFG.yf_suffix}": s for s in batch}
        log.info("  batch %d-%d / %d ...", i + 1, i + len(batch), len(to_fetch))
        try:
            raw = _download_batch(list(ns_map))
        except Exception as exc:  # noqa: BLE001
            log.warning("  batch download error: %s", exc)
            raw = None

        for yf_t, sym in ns_map.items():
            frame = _extract_one(raw, yf_t) if raw is not None else None
            if frame is None or frame.empty:
                failed.append(sym)
                continue
            frame.insert(0, "Symbol", sym)
            frame.to_csv(_cache_path(sym), index=False)
        time.sleep(CFG.yf_pause_seconds)

    # ---- retry the stragglers one-by-one ---------------------------------- #
    if failed:
        log.info("Retrying %d failed tickers individually ...", len(failed))
    still_failed = []
    for sym in failed:
        got = None
        for attempt in range(CFG.yf_max_retries):
            try:
                raw = _download_batch([f"{sym}{CFG.yf_suffix}"])
                got = _extract_one(raw, f"{sym}{CFG.yf_suffix}")
                if got is not None and not got.empty:
                    break
            except Exception as exc:  # noqa: BLE001
                log.debug("   %s retry %d failed: %s", sym, attempt + 1, exc)
            time.sleep(CFG.yf_pause_seconds * (attempt + 1))
        if got is None or got.empty:
            still_failed.append(sym)
        else:
            got.insert(0, "Symbol", sym)
            got.to_csv(_cache_path(sym), index=False)

    if still_failed:
        log.warning("Could not download %d tickers: %s",
                    len(still_failed), ", ".join(still_failed))

    # ---- assemble long-format master from cache --------------------------- #
    frames = []
    for s in symbols:
        p = _cache_path(s)
        if exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise RuntimeError("No price data could be downloaded for ANY symbol.")

    master = pd.concat(frames, ignore_index=True)
    master["Date"] = pd.to_datetime(master["Date"]).dt.date
    master = master.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    save_csv(master, FILES["historical"])
    log.info("Historical master: %d symbols, %d rows.",
             master["Symbol"].nunique(), len(master))
    return master


if __name__ == "__main__":
    from .utils import load_csv
    syms = load_csv(FILES["universe"])["Symbol"].tolist()
    download_prices(syms)
