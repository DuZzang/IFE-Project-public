"""HHZ moving-average sensitivity: the pairs Han, Huang & Zhou (2019, footnote 3) report as robust.

Usage: python src/hhz_ma_sensitivity.py PATH/TO/sp500_monthly.csv
Pre-specified on 2026-09-29 before running; all pairs are reported: MA 50/200 (main), 60/252, 30/100.
HHZ is used as a trend-based exposure rule: a selected stock without a confirmed uptrend (MA(short) > MA(long)) is not held and its
weight earns the T-bill rate (``managed_portfolios``, "HHZ (trend, cash)").
  - Equal 25% product (CNN, MOM, low beta, low IVOL), decile 10 and top 50, EW, net of 0.1%, 2005-01 to 2019-12
  - three-factor product in 2020 (stress check only; no CNN images after the 2019-12 formation)
  - selection test: next-month return of kept minus removed stocks within the selection (2005-2019)
Only portfolio-level aggregates are written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import assign_deciles  # noqa: E402
from eval_window import END, FIRST_FORMATION, START  # noqa: E402
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402
from managed_portfolios import (DESIGNS, KEY, OUTPUT_DIR, Daily, bsc_replication, monthly_long, run_product,  # noqa: E402
                                summary_row, versus_base, year_2020)
from performance_metrics import newey_west_t_stat  # noqa: E402
from three_factor_combination import build_panel, combined_score, pct_rank  # noqa: E402
from top_n_combination import CNN_SCORES  # noqa: E402

PAIRS = [(50, 200), (60, 252), (30, 100)]
HHZ = [d for d in DESIGNS if d[0] in ("Base", "HHZ (trend, cash)")]


def ranked(panel: pd.DataFrame, with_cnn: bool) -> tuple[pd.DataFrame, dict[str, float]]:
    g = panel.groupby("formation_month")
    frame = panel.assign(mom_rank=g["mom"].transform(pct_rank), beta_rank=1 - g["beta"].transform(pct_rank),
                         ivol_rank=1 - g["ivol_daily"].transform(pct_rank))
    base = {"mom": 1 / 3, "beta_rank": 1 / 3, "ivol_rank": 1 / 3}
    if with_cnn:
        frame["cnn_rank"] = g["cnn_prob_up"].transform(pct_rank)
        base = {"mom": 0.25, "beta_rank": 0.25, "ivol_rank": 0.25, "cnn_rank": 0.25}
    return frame.drop(columns="mom").rename(columns={"mom_rank": "mom"}), base


def selection_test(frame: pd.DataFrame, base: dict[str, float], trend: pd.DataFrame) -> dict:
    scored = frame.assign(score=combined_score([frame[c] for c in base], list(base.values())))
    scored = scored.merge(trend, on=KEY, how="left")
    scored["no_uptrend"] = scored["no_uptrend"].fillna(True).astype(bool)
    chosen = {"D10": scored.loc[assign_deciles(scored, signal_column="score").astype(int) == 10],
              "Top 50": scored.sort_values(["formation_month", "score", "PERMNO"], ascending=[True, False, True])
              .groupby("formation_month").head(50)}
    out = {}
    for name, h in chosen.items():
        h = h.dropna(subset=["ret_final"])
        m = h.groupby(["formation_month", "no_uptrend"])["ret_final"].mean().unstack()
        diff = (m[False] - m[True]).dropna() if True in m else pd.Series(dtype=float)
        out[name] = {"removed_share": h["no_uptrend"].mean(), "kept_minus_removed": 12 * diff.mean(),
                     "kept_minus_removed_t": newey_west_t_stat(diff, lag=4)}
    return out


def main(sp_path: str) -> None:
    sp = pd.read_csv(sp_path, dtype={"month": str}).set_index("month")["sp500tr_ret"]
    rf = load_french_monthly(FF5_MONTHLY_ZIP)["rf"]
    rf.index = rf.index.astype(str)
    daily = Daily()
    _, _, wml_sigma = bsc_replication(daily)
    scores = pd.read_csv(CNN_SCORES, usecols=[*KEY, "cnn_prob_up"], dtype={"formation_month": str})
    four, four_base = ranked(build_panel()[0].merge(scores, on=KEY, how="inner", validate="one_to_one"), True)
    three, three_base = ranked(build_panel(FIRST_FORMATION, "2020-11")[0], False)

    rows, monthly_rows = [], []
    for short, long in PAIRS:
        main_run = run_product(four, four_base, wml_sigma["EW"], daily, rf, END, ma=(short, long), designs=HHZ)
        stress = run_product(three, three_base, wml_sigma["EW"], daily, rf, "2020-12", ma=(short, long), designs=HHZ)
        selection = selection_test(four.loc[four["formation_month"] <= "2019-11"], four_base, daily.trend(short, long))
        monthly_rows += [monthly_long(main_run).assign(ma=f"{short}/{long}", sample="Equal 25%"),
                         monthly_long(stress).assign(ma=f"{short}/{long}", sample="3 signals")]
        for (portfolio, label), monthly in main_run.items():
            if label == "Base" and (short, long) != PAIRS[0]:
                continue
            year = year_2020(stress[(portfolio, label)]["net"], stress[(portfolio, label)]["invested"])
            rows.append({"ma": "—" if label == "Base" else f"{short}/{long}", **summary_row(portfolio, label, monthly, rf, sp),
                         **versus_base(main_run, portfolio, label, rf),
                         **{f"2020_{k}": v for k, v in year.items()},
                         **({} if label == "Base" else selection[portfolio])})
    table = pd.DataFrame(rows).sort_values("portfolio", kind="stable")  # base first, then the pairs in PAIRS order
    table.to_csv(OUTPUT_DIR / "hhz_ma_sensitivity.csv", index=False)
    pd.concat(monthly_rows).to_csv(OUTPUT_DIR / "hhz_ma_sensitivity_monthly.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["portfolio", "ma", "avg_invested", "turnover_one_way", "annual_mean", "sharpe", "max_drawdown", "year_2008",
            "mar_may_2009", "vs_base_sharpe_diff", "vs_base_ci95_low_6m", "vs_base_ci95_high_6m", "2020_year_2020",
            "2020_invested_04", "kept_minus_removed", "kept_minus_removed_t"]
    print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
