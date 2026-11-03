"""Does Barroso & Santa-Clara's premise hold in our data, and do BSC and DM act through it? Saved aggregates only.

Usage: python src/bsc_validity.py   (after managed_portfolios.py)
Writes outputs/managed_portfolios/bsc_validity.csv. Checks, 2005-01..2019-12 unless noted:
  a  persistence: next-month squared EW WML return on the variance forecast (slope, Newey-West t, R2)
  b  EW WML by tercile of the ex-ante volatility forecast (mean, volatility, raw Sharpe without RF, worst month)
  c  DM (appendix): BSC-A-style momentum weight timing, (DM - Base) net return on WML and WML x (k - 1)
  d  BSC: Base returns in months when BSC cut exposure (e < 1) vs the other months
Sharpe ratios here are mean / std of raw returns (no risk-free rate) and the groups are small (29-75 months).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from managed_portfolios import OUTPUT_DIR, TARGET  # noqa: E402


def nw_ols(y: np.ndarray, X: np.ndarray, lag: int = 4) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with a constant; slopes, Newey-West (Bartlett) t-stats and R2."""
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    bread = np.linalg.inv(X.T @ X)
    s = X * e[:, None]
    meat = s.T @ s
    for k in range(1, lag + 1):
        c = s[k:].T @ s[:-k]
        meat += (1 - k / (lag + 1)) * (c + c.T)
    return b, b / np.sqrt(np.diag(bread @ meat @ bread)), 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))


def stats(r: pd.Series) -> dict:
    return {"months": len(r), "mean_ann": 12 * r.mean(), "vol_ann": math.sqrt(12) * r.std(),
            "sharpe_raw": math.sqrt(12) * r.mean() / r.std(), "worst_month": r.min()}


def main() -> None:
    w = pd.read_csv(OUTPUT_DIR / "bsc_wml_monthly.csv", dtype={"holding_month": str}).set_index("holding_month")
    w = w.dropna(subset=["weight EW"]).assign(sig=lambda d: TARGET / d["weight EW"])
    monthly = pd.read_csv(OUTPUT_DIR / "product_managed_monthly.csv", dtype={"holding_month": str})
    net = monthly.pivot(index="holding_month", columns="series", values="net")
    invested = monthly.pivot(index="holding_month", columns="series", values="invested")
    rows = []
    for first, window in [("2005-01", "2005-2019"), ("2001-01", "2001-2019")]:
        x = w.loc[first:"2019-12"]
        b, t, r2 = nw_ols((x["WML EW"] ** 2).to_numpy(), (x["sig"] ** 2).to_numpy())
        rows.append({"check": "a", "window": window, "group": "all", "months": len(x), "slope": b[1], "t": t[1], "r2": r2})
        tercile = pd.qcut(x["sig"], 3, labels=["low", "mid", "high"])
        for group, r in x.groupby(tercile, observed=True)["WML EW"]:
            rows.append({"check": "b", "window": window, "group": f"sigma forecast {group}", **stats(r)})
    x = w.loc["2005-01":"2019-12"]
    for portfolio in ["D10", "Top 50"]:
        diff = (net[f"{portfolio} | DM (momentum weight)"] - net[f"{portfolio} | Base"]).loc[x.index]
        b, t, r2 = nw_ols(diff.to_numpy(), np.column_stack([x["WML EW"], x["WML EW"] * (x["weight EW"] - 1)]))
        low = x["weight EW"] < 1
        rows.append({"check": "c", "window": "2005-2019", "group": portfolio, "months": len(x), "slope": b[2], "t": t[2], "r2": r2,
                     "corr_k_below_1": np.corrcoef(diff[low], x.loc[low, "WML EW"])[0, 1],
                     "corr_k_at_least_1": np.corrcoef(diff[~low], x.loc[~low, "WML EW"])[0, 1]})
        e = invested[f"{portfolio} | BSC (portfolio volatility)"].loc["2005-01":"2019-12"]
        base = net[f"{portfolio} | Base"].loc[e.index]
        for group, mask in [("BSC cut exposure (e<1)", e < 0.999), ("BSC fully invested (e=1)", e >= 0.999)]:
            rows.append({"check": "d", "window": "2005-2019", "group": f"{portfolio}: {group}", **stats(base[mask])})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "bsc_validity.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
