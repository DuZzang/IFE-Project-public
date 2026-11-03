"""BSC with leverage allowed (BSC = the design called BSC-B before 2026-09-30): Barroso & Santa-Clara (2015) eq. 6 as written, exposure = target / sigma_hat.

Usage: python src/leverage_bsc.py PATH/TO/sp500_monthly.csv
Decided 2026-09-30 (the team has no leverage restriction); pre-specified before running, every row reported.
The product is the Equal 25% four-signal long-only D10 / top 50 (EW, net of 0.1% per dollar traded), 2005-01 to
2019-12, and the three-signal version through 2020 as a stress check. Designs:
  Base; BSC with exposure cap 1 (the representative, no leverage), 1.5, 2 (Reg T initial margin) and none (the paper);
  HHZ (trend, cash); BSC + HHZ with cap 1, 2 and none.
Exposure above 1 is borrowed at the FF risk-free rate (``managed_returns``); a borrowing spread of 1% a year over RF
is a sensitivity (``*_spread1`` columns). The volatility forecast is the paper's eq. 5 on the unscaled portfolio's
daily path. Maintenance-margin check: within each levered month, the lowest equity / stock value of the levered
Base portfolio (25% is the usual maintenance requirement). Only portfolio-level aggregates are written.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_window import END, FIRST_FORMATION, START  # noqa: E402
from hhz_ma_sensitivity import ranked  # noqa: E402
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402
from longonly_d10_vs_sp500 import metrics  # noqa: E402
from managed_portfolios import (KEY, OUTPUT_DIR, Daily, bsc_replication, code_month, month_code, monthly_long,  # noqa: E402
                                run_product, select_top, summary_row, versus_base, year_2020)
from three_factor_combination import build_panel, combined_score  # noqa: E402
from top_n_combination import CNN_SCORES  # noqa: E402

SPREAD = 0.01  # borrowing spread over RF, per year (sensitivity)
MAINTENANCE = 0.25
DESIGNS = [("Base", 1, None, False), ("BSC cap 1", 1, None, 1.0), ("BSC cap 1.5", 1, None, 1.5),
           ("BSC cap 2", 1, None, 2.0), ("BSC no cap (paper)", 1, None, math.inf), ("HHZ (trend, cash)", 1, "cash", False),
           ("BSC cap 1 + HHZ", 1, "cash", 1.0), ("BSC cap 2 + HHZ", 1, "cash", 2.0),
           ("BSC no cap + HHZ", 1, "cash", math.inf)]


def with_spread(monthly: pd.DataFrame) -> pd.Series:
    return monthly["net"] - SPREAD / 12 * (monthly["invested"] - 1).clip(lower=0)


def leverage_row(monthly: pd.DataFrame, rf: pd.Series, sp: pd.Series) -> dict:
    borrowed = (monthly["invested"] - 1).clip(lower=0)
    spread = metrics(with_spread(monthly), rf.reindex(monthly.index), sp.reindex(monthly.index))
    return {"max_invested": monthly["invested"].max(), "months_levered": int((monthly["invested"] > 1 + 1e-9).sum()),
            "avg_borrowed": borrowed.mean(), **{f"{k}_spread1": spread[k] for k in ["annual_mean", "sharpe", "max_drawdown"]}}


def margin_check(frame: pd.DataFrame, base: dict[str, float], daily: Daily, results: dict, last: str) -> list[dict]:
    """Lowest intramonth equity / stock value of each levered BSC (no HHZ) portfolio; stocks are the Base holdings."""
    frame = frame.assign(mcode=month_code(frame["formation_month"]) + 1,
                         score=combined_score([frame[c] for c in base], list(base.values())))
    rows = []
    for portfolio in ["D10", "Top 50"]:
        chosen = select_top(frame, portfolio)
        members = chosen[["PERMNO", "mcode"]].assign(w=1.0 / chosen.groupby("formation_month")["PERMNO"].transform("size"))
        stock = daily.portfolio(members)  # all weight in stocks: the stock sleeve's daily return
        day_month = daily.day_month[stock.index]
        for label, *_ in DESIGNS:
            if not label.startswith("BSC") or "HHZ" in label:
                continue
            invested = results[(portfolio, label)]["invested"]
            worst, worst_month = np.inf, None
            for month, lev in invested[invested > 1 + 1e-9].items():
                if month > last:
                    continue
                days = day_month == month_code(pd.Series([month]))[0]
                s = np.cumprod(1 + stock.to_numpy()[days])
                c = np.cumprod(1 + daily.rf[stock.index[days]])
                ratio = ((lev * s - (lev - 1) * c) / (lev * s)).min()
                if ratio < worst:
                    worst, worst_month = ratio, month
            rows.append({"portfolio": portfolio, "design": label, "lowest_equity_to_stock": worst,
                         "month": worst_month, "below_maintenance": bool(worst < MAINTENANCE)})
    return rows


def main(sp_path: str) -> None:
    sp = pd.read_csv(sp_path, dtype={"month": str}).set_index("month")["sp500tr_ret"]
    rf = load_french_monthly(FF5_MONTHLY_ZIP)["rf"]
    rf.index = rf.index.astype(str)
    daily = Daily()
    _, _, wml_sigma = bsc_replication(daily)
    scores = pd.read_csv(CNN_SCORES, usecols=[*KEY, "cnn_prob_up"], dtype={"formation_month": str})
    four, four_base = ranked(build_panel()[0].merge(scores, on=KEY, how="inner", validate="one_to_one"), True)
    three, three_base = ranked(build_panel(FIRST_FORMATION, "2020-11")[0], False)

    main_run = run_product(four, four_base, wml_sigma["EW"], daily, rf, END, designs=DESIGNS)
    stress = run_product(three, three_base, wml_sigma["EW"], daily, rf, "2020-12", designs=DESIGNS)
    rows = []
    for (portfolio, label), monthly in main_run.items():
        year = year_2020(stress[(portfolio, label)]["net"], stress[(portfolio, label)]["invested"])
        rows.append({**summary_row(portfolio, label, monthly, rf, sp), **leverage_row(monthly, rf, sp),
                     **versus_base(main_run, portfolio, label, rf), **{f"2020_{k}": v for k, v in year.items()}})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "leverage_bsc.csv", index=False)
    pd.concat([monthly_long(main_run).assign(sample="Equal 25%"), monthly_long(stress).assign(sample="3 signals")]
              ).to_csv(OUTPUT_DIR / "leverage_bsc_monthly.csv", index=False)
    margin = pd.DataFrame(margin_check(four, four_base, daily, main_run, END))
    margin.to_csv(OUTPUT_DIR / "leverage_bsc_margin.csv", index=False)

    pd.set_option("display.width", 260)
    cols = ["portfolio", "design", "avg_invested", "max_invested", "months_levered", "turnover_one_way", "annual_mean",
            "volatility", "sharpe", "max_drawdown", "sharpe_spread1", "max_drawdown_spread1", "vs_base_sharpe_diff",
            "vs_base_ci95_low_6m", "vs_base_ci95_high_6m", "year_2008", "2020_year_2020", "2020_feb_mar_crash",
            "2020_invested_03"]
    print(table[cols].round(3).to_string(index=False))
    print(margin.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
