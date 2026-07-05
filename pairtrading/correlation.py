"""
correlation.py  —  STEP 6: correlation filter
=============================================
Pearson correlation of DAILY RETURNS for every pair, keeping those with
correlation >= `corr_threshold` (0.70).

Implementation note
-------------------
We compute the full N x N correlation matrix in one vectorised C call
(`DataFrame.corr`) and slice the upper triangle.  For N~210 this evaluates all
~22k pairs in well under a second — dramatically faster and more numerically
consistent than a Python multiprocessing loop over pairs.  Multiprocessing is
reserved for the genuinely per-pair, expensive steps (cointegration, backtest).

Output: data/correlated_pairs.csv  [Stock1, Stock2, Correlation]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG, FILES
from .utils import log, save_csv


def correlation_filter(returns: pd.DataFrame,
                       threshold: float = CFG.corr_threshold) -> pd.DataFrame:
    corr = returns.corr(method="pearson")

    # take the strict upper triangle to avoid self-pairs and duplicates
    cols = corr.columns.to_numpy()
    mat = corr.to_numpy()
    iu, ju = np.triu_indices(len(cols), k=1)
    flat = pd.DataFrame(
        {
            "Stock1": cols[iu],
            "Stock2": cols[ju],
            "Correlation": mat[iu, ju],
        }
    )

    kept = (
        flat[flat["Correlation"] >= threshold]
        .sort_values("Correlation", ascending=False)
        .reset_index(drop=True)
    )
    log.info("Correlation filter: %d / %d pairs have corr >= %.2f",
             len(kept), len(flat), threshold)
    save_csv(kept, FILES["correlated"])
    return kept
