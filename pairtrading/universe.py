"""
universe.py  —  STEP 1: obtain the F&O stock universe
=====================================================
Pulls the *live* list of NSE F&O-eligible securities and selects up to 250
names ranked by traded value (a robust liquidity proxy that also correlates
with market-cap for large caps).

Primary source : NSE "Securities in F&O" JSON API (symbols + live prices/volume)
Backup source  : NSE fo_mktlots.csv (symbol list only)
Last resort    : embedded fallback_universe list (network-independent)

Output: data/fno_250_stocks.csv  with columns [Symbol, CompanyName]
        (plus diagnostic columns when live data is available)
"""

from __future__ import annotations

import io

import pandas as pd

from config import CFG, FILES
from .fallback_universe import FALLBACK_FNO_SYMBOLS
from .utils import log, make_nse_session, nse_get_json, save_csv


def _from_nse_api() -> pd.DataFrame | None:
    """Live snapshot: symbol, company name, last price, volume, traded value."""
    sess = make_nse_session(CFG.nse_home, retries=CFG.yf_max_retries)
    if sess is None:
        return None

    payload = nse_get_json(sess, CFG.nse_fno_api)
    if not payload or "data" not in payload:
        log.warning("NSE F&O API returned no usable payload.")
        return None

    rows = []
    for item in payload["data"]:
        sym = str(item.get("symbol", "")).strip()
        # skip index rows / blanks (index queries sometimes prepend the index)
        if not sym or sym.upper().startswith("NIFTY"):
            continue
        meta = item.get("meta") or {}
        last = item.get("lastPrice")
        vol = item.get("totalTradedVolume")
        # traded value (Rs) = price * volume; fall back to API field if present
        tval = item.get("totalTradedValue")
        if tval in (None, 0) and last and vol:
            tval = float(last) * float(vol)
        rows.append(
            {
                "Symbol": sym,
                "CompanyName": (meta.get("companyName") or "").strip() or sym,
                "LastPrice": last,
                "Volume": vol,
                "TradedValue": tval,
            }
        )

    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset="Symbol")
    log.info("NSE API returned %d F&O securities.", len(df))
    return df


def _from_nse_lots_csv() -> pd.DataFrame | None:
    """Backup: the market-lots CSV lists every F&O underlying (symbol only)."""
    sess = make_nse_session(CFG.nse_home, retries=CFG.yf_max_retries)
    if sess is None:
        return None
    try:
        resp = sess.get(CFG.nse_fno_lots_csv, timeout=20)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:  # noqa: BLE001 - any failure -> fall through
        log.warning("fo_mktlots.csv fetch failed: %s", exc)
        return None

    # The file has a 'SYMBOL' (or 'Symbol') column plus lot sizes per expiry.
    col = next((c for c in raw.columns if c.strip().upper() == "SYMBOL"), None)
    if col is None:
        return None
    syms = (
        raw[col].astype(str).str.strip().str.upper()
        .loc[lambda s: ~s.isin(["", "SYMBOL", "NIFTY", "BANKNIFTY", "FINNIFTY",
                                "MIDCPNIFTY", "NIFTYNXT50"])]
        .unique()
        .tolist()
    )
    log.info("fo_mktlots.csv yielded %d underlyings.", len(syms))
    return pd.DataFrame({"Symbol": syms, "CompanyName": syms})


def _from_fallback() -> pd.DataFrame:
    log.warning("Using EMBEDDED fallback F&O universe (%d symbols).",
                len(FALLBACK_FNO_SYMBOLS))
    return pd.DataFrame({"Symbol": FALLBACK_FNO_SYMBOLS,
                         "CompanyName": FALLBACK_FNO_SYMBOLS})


def build_universe(target: int = CFG.target_universe_size) -> pd.DataFrame:
    """
    Selection logic
    ---------------
    1. Prefer the live NSE API (gives price + volume -> liquidity ranking).
    2. If it is down, use fo_mktlots.csv (symbols only), then embedded fallback.
    3. Clean symbols: uppercase, strip, drop obvious non-equity / index rows.
    4. If > target names exist, keep the top `target` by TradedValue (liquidity,
       which for large caps tracks market-cap).  If <= target, keep them all and
       report the true count (NSE presently lists < 250 single-stock F&O names).
    """
    source = "nse_api"
    df = _from_nse_api()
    if df is None or df.empty:
        source = "nse_lots_csv"
        df = _from_nse_lots_csv()
    if df is None or df.empty:
        source = "fallback"
        df = _from_fallback()

    # --- clean symbols ------------------------------------------------------ #
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df = df[df["Symbol"].str.match(r"^[A-Z0-9&\-]+$")]           # valid tickers
    df = df.drop_duplicates(subset="Symbol").reset_index(drop=True)

    # --- rank + trim -------------------------------------------------------- #
    if "TradedValue" in df.columns and df["TradedValue"].notna().any():
        df = df.sort_values("TradedValue", ascending=False, na_position="last")
        rank_basis = "traded value (liquidity)"
    else:
        df = df.sort_values("Symbol")
        rank_basis = "alphabetical (no live liquidity data available)"

    total_available = len(df)
    df = df.head(target).reset_index(drop=True)

    log.info(
        "Universe source=%s | available=%d | selected=%d | ranked by %s",
        source, total_available, len(df), rank_basis,
    )
    if total_available < target:
        log.warning(
            "Only %d F&O single-stock names available (< requested %d) — "
            "using ALL of them (NSE currently lists fewer than 250).",
            total_available, target,
        )

    out_cols = ["Symbol", "CompanyName"]
    diagnostics = [c for c in ("LastPrice", "Volume", "TradedValue") if c in df.columns]
    save_csv(df[out_cols + diagnostics], FILES["universe"])
    return df


if __name__ == "__main__":
    build_universe()
