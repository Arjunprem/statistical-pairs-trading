"""
build_notebook.py
=================
Generates `notebooks/pair_trading_analysis.ipynb` from code (nbformat), so the
analysis notebook is reproducible and version-control friendly.  Run:

    python build_notebook.py            # write the .ipynb
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/pair_trading_analysis.ipynb    # (optional) embed outputs
"""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# --------------------------------------------------------------------------- #
md(r"""
# Indian F&O Pair-Trading — Analysis Notebook

End-to-end walkthrough of the statistical pair-trading pipeline: universe →
data → clean → correlation → cointegration (3 windows) + ADF → hedge ratio →
z-score signals → backtest.

All heavy computation lives in the `pairtrading` package and `run_pipeline.py`;
this notebook **loads the produced artefacts** and visualises them.  If the
`data/` CSVs are missing, run `python run_pipeline.py` first.
""")

code(r"""
# --- setup: make the project importable and point at the artefacts ---------- #
import os, sys
from pathlib import Path

# locate the project root (this notebook lives in notebooks/)
ROOT = Path.cwd()
if (ROOT / "notebooks").exists() and (ROOT / "config.py").exists():
    pass
elif (ROOT.parent / "config.py").exists():
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import CFG, FILES
from pairtrading.returns import load_clean_matrix, compute_returns
from pairtrading.signals import compute_spread, zscore, generate_positions

pd.set_option("display.width", 160)
plt.rcParams["figure.figsize"] = (11, 4.5)
plt.rcParams["axes.grid"] = True
print("Project root:", ROOT)
print("History window:", CFG.start_date, "->", CFG.end_date)
""")

# --------------------------------------------------------------------------- #
md("## 1. Universe & data")
code(r"""
universe = pd.read_csv(FILES["universe"])
clean = load_clean_matrix(FILES["clean"])
print(f"F&O universe selected : {len(universe)} symbols")
print(f"Clean price matrix     : {clean.shape[0]} trading days x {clean.shape[1]} symbols")
universe.head(10)
""")

# --------------------------------------------------------------------------- #
md("## 2. The selection funnel\nHow the ~20k candidate pairs are filtered down.")
code(r"""
counts = {
    "All pairs (NC2)":        len(pd.read_csv(FILES["all_pairs"])),
    "Correlation >= %.2f" % CFG.corr_threshold: len(pd.read_csv(FILES["correlated"])),
    "Cointegrated (gate=%s)" % "+".join(CFG.coint_gate_windows): len(pd.read_csv(FILES["cointegrated"])),
    "ADF-stationary + selected": len(pd.read_csv(FILES["selected"])),
}
funnel = pd.Series(counts)
display(funnel.to_frame("pairs"))

ax = funnel.plot(kind="barh", color="#3b7dd8")
ax.invert_yaxis(); ax.set_title("Pair-selection funnel"); ax.set_xlabel("number of pairs")
for i, v in enumerate(funnel.values):
    ax.text(v, i, f" {v:,}", va="center")
plt.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------- #
md("## 3. Correlation survivors\nDaily-return Pearson correlation (never on raw prices).")
code(r"""
corr = pd.read_csv(FILES["correlated"])
print(f"{len(corr)} pairs with return-correlation >= {CFG.corr_threshold}")
corr.head(15)
""")

# --------------------------------------------------------------------------- #
md("## 4. Cointegration & ADF diagnostics")
code(r"""
coint = pd.read_csv(FILES["cointegrated"])
adf = pd.read_csv(FILES["adf"])
print("Cointegrated (gate passed) — 120d/60d flags shown as robustness diagnostics:")
display(coint)
print("ADF test on the spread (needs p < %.2f for stationarity):" % CFG.adf_pvalue)
display(adf)
""")

# --------------------------------------------------------------------------- #
md("""
## 5. Deep-dive on the selected pair(s)
For each finally-selected pair we visualise: normalised prices, the return
scatter, the spread with its rolling mean, the z-score with entry/exit bands,
and the price-ratio vs its 30/60/150/200-day moving averages (per the strategy
worksheet).
""")
code(r"""
selected = pd.read_csv(FILES["selected"])
hedge = pd.read_csv(FILES["hedge"])
selected = selected.merge(hedge[["Stock1","Stock2","Beta"]], on=["Stock1","Stock2"],
                          how="left", suffixes=("", "_h"))
if "Beta" not in selected.columns:
    selected["Beta"] = selected["Beta_h"]
display(selected)

def analyse_pair(a, b, beta):
    px = clean[[a, b]].dropna()
    A, B = px[a], px[b]

    # 5a. normalised price levels
    fig, ax = plt.subplots()
    (A/A.iloc[0]).plot(ax=ax, label=a)
    (B/B.iloc[0]).plot(ax=ax, label=b)
    ax.set_title(f"{a} vs {b} — normalised prices"); ax.legend(); plt.show()

    # 5b. daily-return scatter
    r = compute_returns(px)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(r[b], r[a], s=6, alpha=0.4)
    ax.set_xlabel(f"{b} daily return"); ax.set_ylabel(f"{a} daily return")
    ax.set_title(f"Return scatter (corr={r[a].corr(r[b]):.2f})"); plt.show()

    # 5c. spread + rolling mean, and 5d. z-score with bands
    spread = compute_spread(A, B, beta)
    zf = zscore(spread, CFG.zscore_window)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    zf["Spread"].plot(ax=ax1, label="spread = A - beta*B")
    zf["RollingMean"].plot(ax=ax1, label=f"{CFG.zscore_window}d mean", color="orange")
    ax1.set_title(f"{a} - {beta:.2f}*{b} spread"); ax1.legend()
    zf["Z"].plot(ax=ax2, color="#444")
    for lvl, c in [(CFG.z_entry,"r"), (-CFG.z_entry,"g"), (0,"k")]:
        ax2.axhline(lvl, color=c, ls="--", lw=1)
    ax2.axhline(CFG.z_stop, color="maroon", ls=":", lw=1)
    ax2.axhline(-CFG.z_stop, color="maroon", ls=":", lw=1)
    ax2.set_title("Rolling z-score with entry (±2) / exit (0) / stop (±3.5) bands")
    plt.tight_layout(); plt.show()

    # 5e. price ratio vs 30/60/150/200-day MAs (strategy-worksheet request)
    ratio = A / B
    fig, ax = plt.subplots()
    ratio.plot(ax=ax, label="price ratio A/B", color="black", lw=1)
    for w in (30, 60, 150, 200):
        ratio.rolling(w).mean().plot(ax=ax, label=f"{w}d MA", lw=1)
    ax.set_title(f"{a}/{b} price ratio vs moving averages"); ax.legend(ncol=3); plt.show()

for r in selected.itertuples(index=False):
    if pd.notna(getattr(r, "Beta", np.nan)):
        analyse_pair(r.Stock1, r.Stock2, float(r.Beta))
""")

# --------------------------------------------------------------------------- #
md("## 6. Backtest — equity curve, drawdown & trades")
code(r"""
def backtest_curve(a, b, beta):
    px = clean[[a, b]].dropna(); A, B = px[a], px[b]
    spread = compute_spread(A, B, beta)
    zf = zscore(spread, CFG.zscore_window)
    pos = generate_positions(zf["Z"]).shift(1).fillna(0.0)   # act next bar
    dA, dB = A.diff(), B.diff()
    gross = A.shift(1).abs() + abs(beta)*B.shift(1).abs()
    ret = pos*(dA - beta*dB)/gross.replace(0, np.nan)
    cost = pos.diff().abs().fillna(pos.abs())*2*(CFG.txn_cost_bps/1e4)
    net = (ret-cost).fillna(0.0)
    equity = (1+net).cumprod()
    dd = equity/equity.cummax()-1
    fig, (a1,a2) = plt.subplots(2,1, figsize=(11,6), sharex=True)
    equity.plot(ax=a1, color="#1a7f37"); a1.set_title(f"{a}/{b} — strategy equity (growth of 1)")
    a1.set_ylabel("growth of 1")
    dd.plot(ax=a2, color="#b3261e"); a2.fill_between(dd.index, dd.values, color="#b3261e", alpha=0.3)
    a2.set_title("drawdown"); a2.set_ylabel("dd")
    plt.tight_layout(); plt.show()

for r in selected.itertuples(index=False):
    if pd.notna(getattr(r, "Beta", np.nan)):
        backtest_curve(r.Stock1, r.Stock2, float(r.Beta))
""")

code(r"""
print("Per-pair backtest summary:")
display(pd.read_csv(FILES["backtest"]))
print("\nTrade log:")
display(pd.read_csv(FILES["signals"]))
""")

# --------------------------------------------------------------------------- #
md("""
## 7. Notes & next steps
- Selection is intentionally strict (correlation ≥ 0.70 on returns, 1-year
  Engle-Granger cointegration gate + ADF stationarity), so few pairs survive —
  this is the honest result on live 2022–2026 data, not a bug.
- To widen the tradable set, lower `CFG.corr_threshold` (e.g. 0.60) or relax the
  cointegration gate in `config.py`, then rerun `python run_pipeline.py --force`.
- Follow-ups flagged in the strategy worksheet: walk-forward optimisation and
  ranking by drawdown / time-to-revert (half-life and drawdown columns already
  support the latter).
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = NB_DIR / "pair_trading_analysis.ipynb"
nbf.write(nb, out)
print("Wrote", out)
