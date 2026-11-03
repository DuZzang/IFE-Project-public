"""Long-only decile 10 portfolios benchmarked against the S&P 500 (professor feedback, 2026-09-29).

Usage: python src/longonly_d10_vs_sp500.py PATH/TO/sp500_monthly.csv [-1]
(the optional number replaces every G/M/V delisting-month return: stress test, suffixed outputs)
The CSV has month-end levels and returns of ^SP500TR and ^GSPC (Yahoo Finance daily closes,
last close of each month). Holding months 2005-01 to 2019-12 (``eval_window``).
Portfolios (buy decile 10 only, EW and VW, delisting-corrected):
  - Vanilla CNN on the CNN universe (deciles re-assigned after the formation tradability filter).
  - On the four-signal matched universe, ranks recomputed within it: CNN only, 3 signals,
    equal 25%, CNN 40% (20/20/20/40), CNN 70% (10/10/10/70); also their top 50 (EW).
Stocks already delisting in the formation month are excluded before ranking.
Gross and net of 0.10% per dollar traded (drift-adjusted weights, ``top_n_returns``).
Metrics use excess returns over RF; beta, alpha, active return, tracking error and information
ratio are measured against the S&P 500 total-return index. Only portfolio-level aggregates are written.
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
from common import assign_deciles  # noqa: E402
from delisting_robustness import cnn_universe  # noqa: E402
from eval_window import clip  # noqa: E402
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402
from performance_metrics import maximum_drawdown, newey_west_t_stat, ols_alpha_hac  # noqa: E402
from three_factor_combination import build_panel, combined_score, pct_rank  # noqa: E402
from top_n_combination import CNN_SCORES, top_n_returns  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/longonly_d10_vs_sp500"
COST = 0.001
ALL = 10**7
DESIGNS = {  # weights on (mom, beta, ivol, cnn) ranks
    "CNN only (matched)": (0.0, 0.0, 0.0, 1.0),
    "3 signals": (1 / 3, 1 / 3, 1 / 3, 0.0),
    "Equal 25%": (0.25, 0.25, 0.25, 0.25),
    "CNN 40%": (0.2, 0.2, 0.2, 0.4),
    "CNN 70%": (0.1, 0.1, 0.1, 0.7),
}


def metrics(r: pd.Series, rf: pd.Series, sp: pd.Series) -> dict:
    x, b = r - rf, sp - rf
    fit = ols_alpha_hac(x.reset_index(drop=True), b.to_frame("sp").reset_index(drop=True), lag=4)
    active = r - sp
    same = active.std(ddof=1) < 1e-10  # a series against itself (the benchmark row): no active risk, t and IR undefined
    return {"annual_mean": 12 * r.mean(), "cagr": float(np.prod(1 + r)) ** (12 / len(r)) - 1,
            "volatility": math.sqrt(12) * r.std(ddof=1), "sharpe": math.sqrt(12) * x.mean() / x.std(ddof=1),
            "max_drawdown": maximum_drawdown(r), "wealth": float(np.prod(1 + r)),
            "beta_sp": float(np.polyfit(b, x, 1)[0]), "alpha_sp": fit["annualized_alpha"],
            "alpha_sp_t": np.nan if same else fit["alpha_t_stat_newey_west"],
            "active_return": 12 * active.mean(), "tracking_error": math.sqrt(12) * active.std(ddof=1),
            "info_ratio": np.nan if same else math.sqrt(12) * active.mean() / active.std(ddof=1),
            "active_t": np.nan if same else newey_west_t_stat(active, lag=4)}


def main(sp_path: str, incomplete_delisting_return: float | None = None) -> None:
    """``incomplete_delisting_return`` (e.g. -1) sets every G/M/V delisting-month return to that value (stress test)."""
    suffix = "" if incomplete_delisting_return is None else f"_gmv_{incomplete_delisting_return:+.0%}"
    sp = pd.read_csv(sp_path, dtype={"month": str}).set_index("month")
    rf = load_french_monthly(FF5_MONTHLY_ZIP)["rf"]
    rf.index = rf.index.astype(str)
    scores = pd.read_csv(CNN_SCORES, usecols=["PERMNO", "formation_month", "cnn_prob_up"], dtype={"formation_month": str})

    legs = {}
    cnn = cnn_universe(incomplete_delisting_return)
    for w in ["EW", "VW"]:
        legs[("Vanilla CNN (CNN universe)", w)] = top_n_returns(cnn.loc[cnn["cnn_decile"] == 10], "cnn_prob_up", ALL, w)[0]

    panel, *_ = build_panel(incomplete_delisting_return=incomplete_delisting_return)
    four = panel.merge(scores, on=["PERMNO", "formation_month"], how="inner", validate="one_to_one")
    g = four.groupby("formation_month")
    ranks = np.column_stack([g["mom"].transform(pct_rank), 1.0 - g["beta"].transform(pct_rank),
                             1.0 - g["ivol_daily"].transform(pct_rank), g["cnn_prob_up"].transform(pct_rank)])
    for name, weights in DESIGNS.items():
        scored = four.assign(score=combined_score(list(ranks.T), weights))
        d10 = scored.loc[assign_deciles(scored, signal_column="score").astype(int) == 10]
        for w in ["EW", "VW"]:
            legs[(name, w)] = top_n_returns(d10, "score", ALL, w)[0]
        legs[(name, "top50 EW")] = top_n_returns(scored, "score", 50, "EW")[0]

    rows = []
    for (name, w), monthly in legs.items():
        m = clip(monthly)
        m.index = m.index.astype(str)
        for label, r in [("gross", m["gross"]), ("net 0.10%", m["gross"] - COST * m["traded"])]:
            rows.append({"portfolio": name, "weighting": w, "returns": label, "months": len(r),
                         "turnover_one_way": m["traded"].mean() / 2, "stocks": m["holdings"].mean(),
                         "missing_held": int(m["missing_returns"].sum()),
                         **metrics(r, rf.reindex(r.index), sp["sp500tr_ret"].reindex(r.index))})
    for col, label in [("sp500tr_ret", "S&P 500 TR"), ("gspc_ret", "S&P 500 price")]:
        r = clip(sp[col])
        rows.append({"portfolio": label, "weighting": "index", "returns": "index", "months": len(r),
                     **metrics(r, rf.reindex(r.index), sp["sp500tr_ret"].reindex(r.index))})
    result = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / f"longonly_d10_vs_sp500{suffix}.csv", index=False)
    pd.DataFrame({f"{n} | {w}": clip(m["gross"] - COST * m["traded"]) for (n, w), m in legs.items()}).assign(
        sp500tr=clip(sp["sp500tr_ret"])).to_csv(OUTPUT_DIR / f"longonly_d10_monthly_net{suffix}.csv")
    pd.set_option("display.width", 260)
    show = ["portfolio", "weighting", "returns", "stocks", "turnover_one_way", "annual_mean", "cagr", "volatility", "sharpe",
            "max_drawdown", "beta_sp", "alpha_sp", "alpha_sp_t", "active_return", "tracking_error", "info_ratio"]
    print(result[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], *(float(x) for x in sys.argv[2:3]))
