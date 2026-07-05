"""
utils.py
========
Shared helpers: logging, an NSE-aware HTTP session, and small IO utilities.

NSE actively blocks non-browser traffic.  The trick that works reliably is:
  1. create a requests.Session with realistic browser headers,
  2. GET the NSE homepage first so the server hands us the anti-bot cookies,
  3. re-use that primed session for the JSON/CSV API calls.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import LOG_DIR


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "pairtrading") -> logging.Logger:
    """Return a configured logger that writes to console + a rotating file."""
    logger = logging.getLogger(name)
    if logger.handlers:                       # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


log = get_logger()


# --------------------------------------------------------------------------- #
# NSE session
# --------------------------------------------------------------------------- #
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}


def make_nse_session(home_url: str, retries: int = 3, timeout: int = 15) -> Optional[requests.Session]:
    """Build a cookie-primed session for the NSE website. Returns None on failure."""
    sess = requests.Session()
    sess.headers.update(_BROWSER_HEADERS)
    for attempt in range(1, retries + 1):
        try:
            # Priming hits — establish the anti-bot cookies.
            sess.get(home_url, timeout=timeout)
            sess.get(home_url + "/market-data/live-equity-market", timeout=timeout)
            if sess.cookies:
                log.info("NSE session primed (%d cookies).", len(sess.cookies))
                return sess
        except requests.RequestException as exc:
            log.warning("NSE priming attempt %d/%d failed: %s", attempt, retries, exc)
            time.sleep(2 * attempt)
    log.error("Could not establish an NSE session after %d attempts.", retries)
    return None


def nse_get_json(sess: requests.Session, url: str, retries: int = 3, timeout: int = 15):
    """GET a JSON endpoint through a primed NSE session, with retries."""
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("NSE JSON attempt %d/%d failed (%s): %s", attempt, retries, url, exc)
            time.sleep(2 * attempt)
    return None


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    log.info("Saved %s  (%d rows x %d cols)", path.name, len(df), df.shape[1])


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0
