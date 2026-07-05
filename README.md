# Indian F&O Statistical Pair-Trading Pipeline

A production-grade, end-to-end pipeline that builds a statistical **pair-trading**
strategy on the NSE Futures & Options (F&O) single-stock universe. It fetches the
universe and price history automatically from the web, funnels ~22k candidate
pairs through correlation → cointegration → stationarity filters, estimates hedge
ratios, generates z-score signals, and backtests every surviving pair.

> Author: **Arjun Premkumar** · Python 3.11+ (tested on 3.14)

---

## Headline result (live NSE data, 2022–2026)

| Stage | Outcome |
|-------|---------|
| F&O universe (pulled live from NSE) | **210** single-stock names |
| Candidate pairs (¹⁹⁸C₂) | 19,503 |
| Survive return-correlation ≥ 0.70 | 22 |
| Cointegrated (1-yr Engle-Granger gate) + ADF-stationary | **1 → `BANKBARODA / CANBK`** |
| Backtest | β 1.54 · **Sharpe 0.87** · **+28.7%** total (6.7% ann.) · −11.3% max DD · 17 trades · **88% win** · ~23-day half-life |

Selection is deliberately strict, so few pairs survive — this is the honest
result on live data, not a bug. Widen it by lowering `CFG.corr_threshold` (e.g.
0.60) or relaxing the cointegration gate in `config.py`, then rerun. See §5.

> **Note on data files:** the heavy raw/intermediate data
> (`data/historical_prices.csv`, `data/clean_prices.csv`) and `cache/` are
> git-ignored — they are regenerated automatically by `python run_pipeline.py`,
> which re-downloads from Yahoo Finance. The small **result** CSVs are committed
> so you can inspect the outcomes without running anything.

---

## 1. Quick start

```bash
# from the pair-trading/ directory
python -m pip install -r requirements.txt

python run_pipeline.py            # run the whole thing (caches each stage)
python run_pipeline.py --force    # recompute every stage from scratch
python run_pipeline.py --from clean   # rerun from the 'clean' stage onward
```

All artefacts land in `data/`. Raw per-ticker downloads are cached in `cache/`
so reruns are near-instant. Logs stream to console and `logs/pipeline.log`.

To explore results interactively:

```bash
jupyter lab notebooks/pair_trading_analysis.ipynb
```

---

## 2. Repository layout

```
pair-trading/
├── config.py                 # ALL tunable parameters (single source of truth)
├── run_pipeline.py           # orchestrator: STEPS 1-12, stage caching
├── requirements.txt
├── README.md
├── pairtrading/              # modular library, one module per pipeline stage
│   ├── utils.py              # logging, NSE session, IO helpers
│   ├── fallback_universe.py  # embedded F&O list (network-independent fallback)
│   ├── universe.py           # STEP 1  – F&O universe from NSE
│   ├── download.py           # STEP 2  – Yahoo Finance OHLCV download
│   ├── clean.py              # STEP 3  – align / clean into a price matrix
│   ├── pairs.py              # STEP 4  – all N-choose-2 pairs
│   ├── returns.py            # STEP 5  – daily returns
│   ├── correlation.py        # STEP 6  – Pearson correlation filter (>= 0.70)
│   ├── cointegration.py      # STEP 7+8 – Engle-Granger (3 windows) + ADF
│   ├── hedge.py              # STEP 9  – OLS hedge ratio + half-life
│   ├── signals.py            # STEP 10 – spread, rolling z-score, entry/exit
│   └── backtest.py           # STEP 11 – event backtest, metrics, trade log
├── notebooks/
│   └── pair_trading_analysis.ipynb
├── data/     # 11 CSV outputs (below)
├── cache/    # per-ticker raw downloads
├── reports/  # plots
└── logs/
```

---

## 3. The 12 steps (methodology)

### STEP 1 — Universe (`universe.py`)
Fetches the live NSE F&O single-stock list. Source priority:
1. **NSE "Securities in F&O" JSON API** — symbols + live price/volume (enables a
   liquidity ranking).
2. **NSE `fo_mktlots.csv`** — the market-lots file (all F&O underlyings).
3. **Embedded fallback list** — used only if NSE is unreachable, so the pipeline
   never dead-ends or requires a manual list.

NSE blocks non-browser traffic, so we prime a `requests.Session` with browser
headers and hit the homepage first to collect the anti-bot cookies.

**Selection logic:** if more than 250 names exist, keep the **top 250 by traded
value** (a liquidity proxy that tracks market-cap for large caps). NSE currently
lists **fewer than 250** single-stock F&O names (~210), so we use **all of them**
and report the true count rather than padding the list with illiquid junk.
→ `fno_250_stocks.csv` `[Symbol, CompanyName]`

### STEP 2 — Historical data (`download.py`)
Downloads ~4 years of **daily** OHLCV per symbol from Yahoo Finance (`.NS`
suffix). `auto_adjust=False` keeps **both** raw `Close` and `Adj Close`. Batched
downloads, per-ticker CSV caching, and per-ticker retries make it robust to
flaky API calls. → `historical_prices.csv`
`[Symbol, Date, Open, High, Low, Close, Adj Close, Volume]`

### STEP 3 — Cleaning (`clean.py`)
Pivots to a wide **Adjusted-Close** matrix (dates × symbols) and enforces:
align on a common calendar; drop duplicate dates; invalidate non-positive
prices; forward-fill only *small* gaps (≤3 days); drop stocks with >10% missing
or <260 valid days; drop residually-sparse dates. → `clean_prices.csv`

### STEP 4 — Pairs (`pairs.py`)
All `N-choose-2` combinations of the cleaned universe. → `all_pairs.csv`
`[Stock1, Stock2]`

### STEP 5 — Returns (`returns.py`)
Daily simple returns `R_t = (P_t − P_{t−1}) / P_{t−1}` from the **clean** prices.
All correlation/statistics use **returns**, never raw price levels (correlation
on trending price levels is spurious).

### STEP 6 — Correlation filter (`correlation.py`)
Pearson correlation of daily returns for every pair; keep **corr ≥ 0.70**.
Computed as a single vectorised `DataFrame.corr()` (full N×N matrix in C), which
evaluates all ~22k pairs in <1s — faster and more consistent than a Python
multiprocessing loop. → `correlated_pairs.csv` `[Stock1, Stock2, Correlation]`

### STEP 7 — Cointegration (`cointegration.py`)
Engle-Granger test on **price levels** over three trailing windows —
**1yr (252d), 120d, 60d**. A pair passes only if `p < 0.05` in **all three**.
Parallelised with a process pool (each worker loads the price matrix once).
→ `cointegrated_pairs.csv`
`[Stock1, Stock2, Cointegration_1yr, Cointegration_120d, Cointegration_60d]`

### STEP 8 — ADF (`cointegration.py`)
Augmented Dickey-Fuller test on each pair's spread (`A − β·B`, β from full-sample
OLS). Stationary spread requires `p < 0.05`.
→ `adf_results.csv` `[Stock1, Stock2, ADF_statistic, ADF_pvalue]`

### STEP 9 — Hedge ratio (`hedge.py`)
OLS `A = α + β·B` (statsmodels; identical to `sklearn.LinearRegression`, which
lacks a Python-3.14 wheel). Spread `= A − β·B`. Also computes the mean-reversion
**half-life** via an AR(1) fit. → `hedge_ratio.csv`
`[Stock1, Stock2, Alpha, Beta, HalfLife]`

### STEP 10 — Z-score & signals (`signals.py`)
`Z = (Spread − RollingMean) / RollingStd` (60-day window). Logic:
`Z ≥ +2` → short spread; `Z ≤ −2` → long spread; exit as `Z` reverts to 0; hard
stop at `|Z| ≥ 3.5`. Signals are generated causally and lagged one bar in the
backtest (no look-ahead).

### STEP 11 — Backtest (`backtest.py`)
Dollar-neutral spread portfolio (`+1` A, `−β` B), returns normalised by gross
notional, **4 bps/leg** transaction costs. Per pair: cumulative return,
annualised **Sharpe**, **max drawdown**, **win %**, **number of trades**, and
convergence-time (holding-day) stats. Parallelised across pairs.
→ `trade_signals.csv` (one row per trade), `backtest_results.csv` (per-pair).

### STEP 12 — Outputs
The **`selected_pairs.csv`** set = pairs passing **correlation ∩ cointegration
(all 3 windows) ∩ ADF**, enriched with correlation, β, half-life and ADF p-value.

---

## 4. Output files (`data/`)

| # | File | Description |
|---|------|-------------|
| 1 | `fno_250_stocks.csv`   | Selected F&O universe |
| 2 | `historical_prices.csv`| Raw daily OHLCV (long format) |
| 3 | `clean_prices.csv`     | Cleaned Adj-Close matrix |
| 4 | `all_pairs.csv`        | Every candidate pair |
| 5 | `correlated_pairs.csv` | Pairs with corr ≥ 0.70 |
| 6 | `cointegrated_pairs.csv`| Cointegrated in all 3 windows |
| 7 | `adf_results.csv`      | ADF statistic & p-value |
| 8 | `hedge_ratio.csv`      | α, β, half-life |
| 9 | `selected_pairs.csv`   | Final tradable set |
| 10| `trade_signals.csv`    | Individual trades |
| 11| `backtest_results.csv` | Per-pair performance |

---

## 5. Key assumptions & design choices

- **Universe < 250:** NSE lists ~210 single-stock F&O names today; we use all of
  them (never pad with illiquid symbols) and report the real count.
- **4 years of history** (spec minimum is 2) gives the 1-yr cointegration window
  plus 60-day rolling z-scores clean buffer.
- **Cointegration runs only on correlation-survivors** — the standard funnel;
  avoids ~90k needless statistical tests without changing the result set.
- **statsmodels OLS** for the hedge ratio (mathematically identical to
  `sklearn.LinearRegression`; sklearn has no 3.14 wheel yet — see requirements).
- **Vectorised correlation**, multiprocessing where it actually pays
  (cointegration, backtest).
- **No look-ahead:** signals are lagged one bar; rolling stats use only trailing
  data.

## 6. Limitations & next steps
- Yahoo Finance adjusted prices are the data of record; corporate-action edge
  cases can differ from exchange feeds.
- In-sample backtest (no walk-forward split yet). The xlsx notes **walk-forward
  optimisation** and **ranking by drawdown / time-to-revert** as follow-ups —
  the half-life and per-pair drawdown columns already support that ranking.
- Costs are modelled as fixed bps; no slippage/impact or F&O lot-size rounding.
- Not investment advice; for research use.
