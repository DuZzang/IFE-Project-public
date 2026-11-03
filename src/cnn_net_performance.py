"""Vanilla CNN after trading costs: long-short D10 - D1, long-only D10, and long-only top 50.

Holding months 2005-01 to 2019-12 (``eval_window``). Traded weight per month = sum of absolute
weight changes against the previous month's drifted weights (``top_n_combination.top_n_returns``).
Long-short net = D10 gross - D1 gross - cost x (traded D10 + traded D1); no short-borrow fee.
Long-only net = gross - cost x traded. Benchmarks: FF short-term reversal (long-short) and the
Fama-French market (long-only; S&P 500 is not stored locally). Only portfolio-level aggregates are written.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "handoff" / "ivol_combination" / "src"))
from delisting_robustness import cnn_universe  # noqa: E402
from eval_window import clip  # noqa: E402
from ivol_specifications import FACTOR_DIR, FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402
from performance_metrics import maximum_drawdown, newey_west_t_stat, ols_alpha_hac  # noqa: E402
from top_n_combination import top_n_returns  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/cnn_net_performance"
COSTS = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005]
ALL = 10**7  # take every stock of a decile leg


def summarize(s: pd.Series, factors: pd.DataFrame, long_only: bool) -> dict:
    f = factors.reindex(s.index)
    x = s - f["rf"] if long_only else s
    bench = ["mkt_rf"] if long_only else ["st_rev"]
    fit = ols_alpha_hac(x.reset_index(drop=True), f[bench].reset_index(drop=True), lag=4)
    return {"annual_mean": 12 * s.mean(), "volatility": math.sqrt(12) * s.std(ddof=1),
            "sharpe": math.sqrt(12) * x.mean() / x.std(ddof=1), "nw_t": newey_west_t_stat(x, lag=4),
            "max_drawdown": maximum_drawdown(s), "beta": float(np.polyfit(f[bench[0]], x, 1)[0]),
            "alpha": fit["annualized_alpha"], "alpha_t": fit["alpha_t_stat_newey_west"]}


def main() -> None:
    factors = load_french_monthly(FF5_MONTHLY_ZIP).join(
        load_french_monthly(FACTOR_DIR / "F-F_ST_Reversal_Factor_CSV.zip"), how="inner")
    factors.index = factors.index.astype(str)
    frame = cnn_universe()

    legs = {}
    for w in ["EW", "VW"]:
        for d in [10, 1]:
            monthly, _ = top_n_returns(frame.loc[frame["cnn_decile"] == d], "cnn_prob_up", ALL, w)
            legs[(w, d)] = clip(monthly)
    top50, _ = top_n_returns(frame, "cnn_prob_up", 50, "EW")
    legs[("EW", "top50")] = clip(top50)

    rows = []
    for cost in COSTS:
        for w in ["EW", "VW"]:
            lo, hi = legs[(w, 10)], legs[(w, 1)]
            ls = lo["gross"] - hi["gross"] - cost * (lo["traded"] + hi["traded"])
            rows.append({"portfolio": "CNN D10-D1", "weighting": w, "cost": cost,
                         "traded_per_month": (lo["traded"] + hi["traded"]).mean(), **summarize(ls, factors, False)})
            d10 = lo["gross"] - cost * lo["traded"]
            rows.append({"portfolio": "CNN D10 long-only", "weighting": w, "cost": cost,
                         "traded_per_month": lo["traded"].mean(), **summarize(d10, factors, True)})
        t50 = legs[("EW", "top50")]
        rows.append({"portfolio": "CNN top 50 long-only", "weighting": "EW", "cost": cost,
                     "traded_per_month": t50["traded"].mean(), **summarize(t50["gross"] - cost * t50["traded"], factors, True)})
    mkt = clip(factors["mkt_rf"] + factors["rf"])
    rows.append({"portfolio": "Market (FF)", "weighting": "VW", "cost": 0.0, **summarize(mkt, factors, True)})
    st = clip(factors["st_rev"])
    rows.append({"portfolio": "ST_Rev (FF)", "weighting": "VW", "cost": 0.0, **summarize(st, factors, False)})

    result = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "cnn_net_performance.csv", index=False)
    pd.set_option("display.width", 250)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
