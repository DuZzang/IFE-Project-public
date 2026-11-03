"""Dynamic rebalancing rules from the teammate package (IFE_dynamic_rebalancing, 2026-09-29) on the official inputs.

Usage: python src/dynamic_rebalancing.py PATH/TO/sp500_monthly.csv
The package built everything from CRSP daily files (approximate signals, FF market as benchmark). This script
reruns its pre-specified rules on the team's official inputs: professor universe tradable at formation, official
signals (Equal 25% with CNN; three signals for the 2020 stress check), ret_final (CRSP monthly delisting returns),
the S&P 500 total-return index. N = 50, or the decile-10 count of the month. EW, net of 0.1% per dollar traded.
  BASE      top N by score, full monthly re-selection
  T_CASH    Han, Huang & Zhou (2019) long-leg rule: BASE names without MA50 > MA200 are sold, weight to T-bills
  T_SCREEN  new names only from uptrend stocks: top N among uptrend stocks
  BAND      holding buffer: last month's names stay while ranked within 2N; the rest is filled by score
  TEAM_DR   the rule written in the 9/29 follow-up PDF (by mistake as HHZ): keep a name ranked within 2N and in
            uptrend; new names only from uptrend stocks
Band multiples 1.5, 3 and 5 are reported as sensitivity, as in the package. Differences from the package: official
signals. The trend test is the package's own (``Daily.trend``, adopted 2026-09-30): no confirmed uptrend = not held.
Sharpe differences against BASE use the paired moving-block bootstrap of ``sharpe_bootstrap``.
Only portfolio-level aggregates are written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_window import END, FIRST_FORMATION, START  # noqa: E402
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402
from managed_portfolios import (KEY, OUTPUT_DIR, Daily, code_month, managed_returns, month_code, monthly_long,  # noqa: E402
                                select_top, summary_row, year_2020)
from sharpe_bootstrap import paired_sharpe_difference  # noqa: E402
from three_factor_combination import build_panel, combined_score, pct_rank  # noqa: E402
from top_n_combination import CNN_SCORES  # noqa: E402

RULES = [("BASE", "base", 2.0), ("T_CASH", "cash", 2.0), ("T_SCREEN", "screen", 2.0), ("BAND", "band", 2.0),
         ("TEAM_DR", "team", 2.0)] + [(f"{name} {m:g}N", kind, m) for m in (1.5, 3.0, 5.0)
                                      for name, kind in [("BAND", "band"), ("TEAM_DR", "team")]]


def select_path(frame: pd.DataFrame, size: str, kind: str, mult: float) -> dict[str, tuple[np.ndarray, int]]:
    """(PERMNOs held, N) for each formation month; rules follow the package's ``dr_backtest.select``.

    Ties follow ``managed_portfolios.select_top``: decile 10 prefers the higher PERMNO (``assign_deciles``), the
    top 50 the lower PERMNO (``top_n_returns``), so BASE equals the other product scripts."""
    out, previous = {}, np.array([], dtype=np.int64)
    for month, cs in frame.groupby("formation_month", sort=True):
        cs = cs.sort_values(["score", "PERMNO"], ascending=[False, size == "Top 50"], kind="mergesort")
        perm, up = cs["PERMNO"].to_numpy(np.int64), ~cs["no_uptrend"].to_numpy(bool)
        pos = np.arange(len(perm))
        n = 50 if size == "Top 50" else int(np.sum(pos * 10 // len(perm) == 9))
        if kind == "base":
            chosen = pos[:n]
        elif kind == "cash":
            chosen = pos[:n][up[:n]]
        elif kind == "screen":
            chosen = pos[up][:n]
        else:
            keep = np.isin(perm, previous) & (pos < round(mult * n)) & (up if kind == "team" else True)
            kept = pos[keep][:n]
            candidates = ~np.isin(pos, kept) & (up if kind == "team" else True)
            chosen = np.concatenate([kept, pos[candidates][: n - len(kept)]])
        out[month] = (perm[chosen], n)
        previous = perm[chosen]
    return out


def run(frame: pd.DataFrame, rf: pd.Series, last: str) -> dict[tuple[str, str], pd.DataFrame]:
    results, months = {}, sorted(frame["holding_month"].unique())
    for size in ["D10", "Top 50"]:
        for label, kind, mult in RULES:
            held = pd.concat([pd.DataFrame({"formation_month": m, "PERMNO": p, "w": 1.0 / n})
                              for m, (p, n) in select_path(frame, size, kind, mult).items()])
            if kind == "base":  # the same BASE as managed_portfolios
                reference = select_top(frame, size)
                assert set(zip(held["formation_month"], held["PERMNO"])) == set(zip(reference["formation_month"], reference["PERMNO"]))
            held = held.merge(frame[[*KEY, "holding_month", "ret_final"]], on=KEY, how="left")
            monthly = managed_returns(held, rf, months)
            results[(size, label)] = monthly.loc[(monthly.index >= START) & (monthly.index <= last)]
    return results


def scored(panel: pd.DataFrame, trend: pd.DataFrame, with_cnn: bool) -> pd.DataFrame:
    g = panel.groupby("formation_month")
    ranks = [g["mom"].transform(pct_rank), 1 - g["beta"].transform(pct_rank), 1 - g["ivol_daily"].transform(pct_rank)]
    if with_cnn:
        ranks.append(g["cnn_prob_up"].transform(pct_rank))
    frame = panel.assign(score=combined_score(ranks, [1 / len(ranks)] * len(ranks)),
                         holding_month=code_month(month_code(panel["formation_month"]) + 1))
    frame = frame.merge(trend, on=KEY, how="left", validate="one_to_one")
    frame["no_uptrend"] = frame["no_uptrend"].fillna(True).astype(bool)
    return frame


def main(sp_path: str) -> None:
    sp = pd.read_csv(sp_path, dtype={"month": str}).set_index("month")["sp500tr_ret"]
    rf = load_french_monthly(FF5_MONTHLY_ZIP)["rf"]
    rf.index = rf.index.astype(str)
    trend = Daily().trend()
    scores = pd.read_csv(CNN_SCORES, usecols=[*KEY, "cnn_prob_up"], dtype={"formation_month": str})
    samples = {"Equal 25% (4 signals)": (scored(build_panel()[0].merge(scores, on=KEY, how="inner"), trend, True), END),
               "3 signals": (scored(build_panel(FIRST_FORMATION, "2020-11")[0], trend, False), "2020-12")}
    rows, monthly_rows = [], []
    for sample, (frame, last) in samples.items():
        results = run(frame, rf, last)
        monthly_rows.append(monthly_long(results).assign(sample=sample))
        for (size, label), monthly in results.items():
            within = monthly.loc[START:END]
            base = results[(size, "BASE")]["net"].loc[START:END]
            row = {"sample": sample, **summary_row(size, label, within, rf, sp),
                   **({} if label == "BASE" else {f"vs_base_{k}": v for k, v in
                                                  paired_sharpe_difference(within["net"], base, rf).items()})}
            if last > END:
                row |= {f"2020_{k}": v for k, v in year_2020(monthly["net"]).items()}
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "dynamic_rebalancing.csv", index=False)
    pd.concat(monthly_rows).to_csv(OUTPUT_DIR / "dynamic_rebalancing_monthly.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["sample", "portfolio", "design", "avg_invested", "turnover_one_way", "annual_mean", "volatility", "sharpe",
            "max_drawdown", "alpha_sp_t", "year_2008", "mar_may_2009", "vs_base_sharpe_diff", "vs_base_ci95_low_6m",
            "vs_base_ci95_high_6m", "2020_year_2020"]
    print(table[[c for c in cols if c in table]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
