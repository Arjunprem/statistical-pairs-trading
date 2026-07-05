"""
enrich_names.py  —  best-effort company-name enrichment for the universe file
=============================================================================
The NSE market-lots file only gives symbols, so `fno_250_stocks.csv` initially
carries CompanyName == Symbol.  This helper backfills real company names from
Yahoo Finance metadata.  It is deliberately best-effort: any failure keeps the
symbol as-is and never blocks the pipeline.  Run it whenever convenient:

    python -m pairtrading.enrich_names
"""

from __future__ import annotations

import warnings

import pandas as pd
import yfinance as yf

from config import CFG, FILES
from .utils import log, save_csv

warnings.simplefilter("ignore")


def enrich_company_names() -> pd.DataFrame:
    df = pd.read_csv(FILES["universe"])
    names = df["CompanyName"].tolist()
    got = 0
    for i, sym in enumerate(df["Symbol"].tolist()):
        # only fetch where the name is still just the symbol
        if str(names[i]).strip().upper() != str(sym).strip().upper():
            continue
        try:
            info = yf.Ticker(f"{sym}{CFG.yf_suffix}").info
            nm = info.get("longName") or info.get("shortName")
            if nm:
                names[i] = nm
                got += 1
        except Exception:  # noqa: BLE001
            pass
        if (i + 1) % 25 == 0:
            log.info("  ...enriched %d/%d (resolved %d)", i + 1, len(df), got)
    df["CompanyName"] = names
    save_csv(df, FILES["universe"])
    log.info("Company names resolved for %d/%d symbols.", got, len(df))
    return df


if __name__ == "__main__":
    enrich_company_names()
