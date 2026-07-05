"""
run_pipeline.py  —  end-to-end orchestrator (STEPS 1-12)
========================================================
Runs the full Indian F&O pair-trading pipeline.  Every stage caches its CSV
artefact, so reruns skip completed work unless ``--force`` is passed.  The
multiprocessing guard (``if __name__ == "__main__"``) is REQUIRED on Windows.

Usage
-----
    python run_pipeline.py                # run everything (use caches)
    python run_pipeline.py --force        # recompute every stage
    python run_pipeline.py --from clean   # rerun from a given stage onward
    python run_pipeline.py --universe 250 # override target universe size
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from config import CFG, FILES
from pairtrading.utils import log, exists, load_csv, save_csv
from pairtrading.universe import build_universe
from pairtrading.download import download_prices
from pairtrading.clean import clean_prices
from pairtrading.pairs import generate_pairs
from pairtrading.returns import compute_returns, load_clean_matrix
from pairtrading.correlation import correlation_filter
from pairtrading.cointegration import run_cointegration
from pairtrading.hedge import compute_hedge_ratios
from pairtrading.backtest import run_backtest

STAGES = ["universe", "download", "clean", "pairs", "correlation",
          "cointegration", "hedge", "backtest"]


def _should_run(stage: str, artefact_key: str, args) -> bool:
    if args.force:
        return True
    if args.from_stage:
        # rerun every stage at/after --from; use cache for earlier ones
        return STAGES.index(stage) >= STAGES.index(args.from_stage)
    return not exists(FILES[artefact_key])


def main() -> None:
    ap = argparse.ArgumentParser(description="Indian F&O pair-trading pipeline")
    ap.add_argument("--force", action="store_true", help="recompute all stages")
    ap.add_argument("--from", dest="from_stage", choices=STAGES, default=None,
                    help="rerun from this stage onward")
    ap.add_argument("--universe", type=int, default=CFG.target_universe_size)
    args = ap.parse_args()

    t0 = time.time()
    log.info("=" * 70)
    log.info("PAIR-TRADING PIPELINE  |  window %s -> %s", CFG.start_date, CFG.end_date)
    log.info("=" * 70)

    # -- STEP 1: universe --------------------------------------------------- #
    if _should_run("universe", "universe", args):
        uni = build_universe(target=args.universe)
    else:
        uni = load_csv(FILES["universe"])
        log.info("[cache] universe: %d symbols", len(uni))
    symbols = uni["Symbol"].tolist()

    # -- STEP 2: history ---------------------------------------------------- #
    if _should_run("download", "historical", args):
        download_prices(symbols, force=args.force)
    else:
        log.info("[cache] historical_prices.csv present")

    # -- STEP 3: clean ------------------------------------------------------ #
    if _should_run("clean", "clean", args):
        long_df = load_csv(FILES["historical"])
        clean = clean_prices(long_df)
    else:
        clean = load_clean_matrix(FILES["clean"])
        log.info("[cache] clean_prices: %d dates x %d symbols", *clean.shape)
    clean_syms = list(clean.columns)

    # -- STEP 4: pairs ------------------------------------------------------ #
    if _should_run("pairs", "all_pairs", args):
        generate_pairs(clean_syms)

    # -- STEP 5 + 6: returns + correlation ---------------------------------- #
    if _should_run("correlation", "correlated", args):
        returns = compute_returns(clean)
        correlated = correlation_filter(returns)
    else:
        correlated = load_csv(FILES["correlated"])
        log.info("[cache] correlated_pairs: %d", len(correlated))

    # -- STEP 7 + 8: cointegration + ADF ------------------------------------ #
    if _should_run("cointegration", "cointegrated", args):
        coint_res = run_cointegration(correlated)
    else:
        coint_res = None
        log.info("[cache] cointegrated_pairs.csv present")

    # -- Build SELECTED set (cointegrated in all 3 windows AND ADF-stationary) #
    if coint_res is not None and not coint_res.empty:
        selected_mask = coint_res["CointGatePass"] & coint_res["ADF_pass"]
        selected = coint_res.loc[selected_mask, ["Stock1", "Stock2", "ADF_pvalue"]].copy()
        selected = selected.merge(correlated, on=["Stock1", "Stock2"], how="left")
    else:
        # rebuild selection from cached artefacts
        coint = load_csv(FILES["cointegrated"])
        adf = load_csv(FILES["adf"])
        good_adf = adf[adf["ADF_pvalue"] < CFG.adf_pvalue][["Stock1", "Stock2", "ADF_pvalue"]]
        selected = coint.merge(good_adf, on=["Stock1", "Stock2"], how="inner")
        corr = load_csv(FILES["correlated"])
        selected = selected.merge(corr, on=["Stock1", "Stock2"], how="left")

    log.info("SELECTED (corr + coint-gate + ADF): %d pairs", len(selected))

    # -- STEP 9: hedge ratios (for the selected set) ------------------------ #
    if len(selected):
        hedge = compute_hedge_ratios(clean, selected[["Stock1", "Stock2"]])
        selected = selected.merge(hedge, on=["Stock1", "Stock2"], how="left")
    else:
        selected["Beta"] = []
    save_csv(selected, FILES["selected"])

    # -- STEP 10 + 11: signals + backtest ----------------------------------- #
    if len(selected):
        results = run_backtest(selected)
        if not results.empty:
            top = results.head(10)
            log.info("Top pairs by Sharpe:\n%s", top.to_string(index=False))
    else:
        log.warning("No pairs survived all filters — nothing to backtest.")
        save_csv(pd.DataFrame(), FILES["signals"])
        save_csv(pd.DataFrame(), FILES["backtest"])

    log.info("=" * 70)
    log.info("PIPELINE COMPLETE in %.1fs.  Artefacts in %s",
             time.time() - t0, FILES["universe"].parent)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
