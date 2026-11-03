"""Long-short (D10 - D1) portfolios benchmarked against the Fama-French short-term reversal factor.

Professor feedback (2026-09-29): a long-short portfolio needs a long-short benchmark, e.g. reversal.
Holding months 2005-01 to 2019-12 (``eval_window``). Portfolios:
  - CNN only: deciles of CNN P(up) (``cnn_universe``: tradable at formation, re-ranked).
  - Combined scores on the four-signal matched universe, ranks recomputed within it:
    3 signals (MOM, low beta, low IVOL, 1/3 each), equal 25%, CNN 40% (20/20/20/40), CNN 70% (10/10/10/70).
Each spread is decile 10 (best) minus decile 1, EW and VW (formation-month cap), delisting-corrected.
Reported: mean, vol, Sharpe (zero-investment, no RF), NW t, max drawdown, correlation with ST_Rev,
alpha/beta on ST_Rev alone, and alpha on FF5 + ST_Rev. Only portfolio-level aggregates are written.
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
from delisting_robustness import cnn_universe, monthly_portfolios  # noqa: E402
from eval_window import clip  # noqa: E402
from ivol_specifications import FACTOR_DIR, FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402
from performance_metrics import maximum_drawdown, newey_west_t_stat, ols_alpha_hac  # noqa: E402
from three_factor_combination import build_panel, combined_score, pct_rank  # noqa: E402
from top_n_combination import CNN_SCORES  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/longshort_vs_reversal"
FF5 = ["mkt_rf", "smb", "hml", "rmw", "cma"]
DESIGNS = {  # weights on (mom, beta, ivol, cnn) ranks
    "3 signals": (1 / 3, 1 / 3, 1 / 3, 0.0),
    "Equal 25%": (0.25, 0.25, 0.25, 0.25),
    "CNN 40%": (0.2, 0.2, 0.2, 0.4),
    "CNN 70%": (0.1, 0.1, 0.1, 0.7),
}


def stats(series: pd.Series, factors: pd.DataFrame) -> dict:
    s = clip(series.dropna())
    s.index = s.index.astype(str)
    f = factors.reindex(s.index)
    rev = ols_alpha_hac(s.reset_index(drop=True), f[["st_rev"]].reset_index(drop=True), lag=4)
    full = ols_alpha_hac(s.reset_index(drop=True), f[FF5 + ["st_rev"]].reset_index(drop=True), lag=4)
    return {
        "months": len(s),
        "annual_mean": 12 * s.mean(),
        "volatility": math.sqrt(12) * s.std(ddof=1),
        "sharpe": math.sqrt(12) * s.mean() / s.std(ddof=1),
        "nw_t": newey_west_t_stat(s, lag=4),
        "max_drawdown": maximum_drawdown(s),
        "corr_st_rev": s.corr(f["st_rev"]),
        "beta_st_rev": float(np.polyfit(f["st_rev"], s, 1)[0]),
        "alpha_st_rev": rev["annualized_alpha"],
        "alpha_st_rev_t": rev["alpha_t_stat_newey_west"],
        "alpha_ff5_st_rev": full["annualized_alpha"],
        "alpha_ff5_st_rev_t": full["alpha_t_stat_newey_west"],
    }


def main() -> None:
    factors = load_french_monthly(FF5_MONTHLY_ZIP).join(
        load_french_monthly(FACTOR_DIR / "F-F_ST_Reversal_Factor_CSV.zip"), how="inner")
    factors.index = factors.index.astype(str)
    series = {}

    cnn = cnn_universe().rename(columns={"cnn_decile": "decile"})
    for w, weighted in [("EW", False), ("VW", True)]:
        wide = monthly_portfolios(cnn, "ret_final", weighted)
        series[("CNN only", w)] = wide[10] - wide[1]

    panel, *_ = build_panel()
    scores = pd.read_csv(CNN_SCORES, usecols=["PERMNO", "formation_month", "cnn_prob_up"], dtype={"formation_month": str})
    four = panel.merge(scores, on=["PERMNO", "formation_month"], how="inner", validate="one_to_one")
    grouped = four.groupby("formation_month")
    ranks = pd.DataFrame({
        "mom": grouped["mom"].transform(pct_rank),
        "beta": 1.0 - grouped["beta"].transform(pct_rank),
        "ivol": 1.0 - grouped["ivol_daily"].transform(pct_rank),
        "cnn": grouped["cnn_prob_up"].transform(pct_rank),
    })
    for name, weights in DESIGNS.items():
        scored = four.assign(score=combined_score([ranks[c] for c in ranks], weights))
        scored = scored.assign(decile=assign_deciles(scored, signal_column="score").astype(int))
        for w, weighted in [("EW", False), ("VW", True)]:
            wide = monthly_portfolios(scored, "ret_final", weighted)
            series[(name, w)] = wide[10] - wide[1]

    rows = [{"portfolio": n, "weighting": w, **stats(s, factors)} for (n, w), s in series.items()]
    st = clip(factors["st_rev"])
    rows.append({"portfolio": "ST_Rev (benchmark)", "weighting": "FF", "months": len(st), "annual_mean": 12 * st.mean(),
                 "volatility": math.sqrt(12) * st.std(ddof=1), "sharpe": math.sqrt(12) * st.mean() / st.std(ddof=1),
                 "nw_t": newey_west_t_stat(st, lag=4), "max_drawdown": maximum_drawdown(st)})
    result = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "longshort_vs_reversal.csv", index=False)
    monthly = pd.DataFrame({f"{n} | {w}": clip(s) for (n, w), s in series.items()})
    monthly["ST_Rev"] = st
    monthly.to_csv(OUTPUT_DIR / "longshort_monthly.csv")
    print("four-signal stock-months:", len(four))
    pd.set_option("display.width", 250)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
