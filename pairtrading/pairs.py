"""
pairs.py  —  STEP 4: generate all candidate pairs
=================================================
Every 2-combination of the cleaned universe.  For N symbols this is N-choose-2.
(N=210 -> 21,945;  the spec's 250 -> 31,125.)

Output: data/all_pairs.csv  with columns [Stock1, Stock2]
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from config import FILES
from .utils import log, save_csv


def generate_pairs(symbols: list[str]) -> pd.DataFrame:
    symbols = sorted(set(symbols))
    pairs = list(combinations(symbols, 2))
    df = pd.DataFrame(pairs, columns=["Stock1", "Stock2"])
    n = len(symbols)
    log.info("Generated %d pairs from %d symbols (expected %d = %dC2).",
             len(df), n, n * (n - 1) // 2, n)
    save_csv(df, FILES["all_pairs"])
    return df


if __name__ == "__main__":
    cols = pd.read_csv(FILES["clean"], nrows=0).columns.drop("Date")
    generate_pairs(list(cols))
