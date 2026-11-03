"""HHZ trend when the price history has gaps: the old rule vs the teammate rule adopted on 2026-09-30.

Usage: python src/hhz_trend_gaps.py
Selected stocks = Base D10 and top 50 of the Equal 25% score, formation months 2004-12..2019-11.
  old rule   MA over the stock's last 50 / 200 valid daily observations; sell only if MA50 < MA200;
             fewer than 200 observations = unknown = keep
  team rule  MA over the last 50 / 200 market days with at least 80% valid days (``Daily.trend``);
             hold only if MA50 > MA200; unknown or flat = sell
Writes outputs/managed_portfolios/hhz_trend_gaps_summary.csv (counts) and hhz_trend_gaps_cases.csv (one row per
stock-month the team rule cannot judge: gap length, how much of the gap is outside the professor CSV, and the
CRSP monthly status in the gap months). No stock identifiers are written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hhz_ma_sensitivity import ranked  # noqa: E402
from managed_portfolios import KEY, OUTPUT_DIR, Daily, month_code, select_top  # noqa: E402
from momentum_lowbeta_deciles import CRSP_MONTHLY  # noqa: E402
from professor_universe import load_professor_stock_data, official_formation_universe  # noqa: E402
from three_factor_combination import build_panel, combined_score  # noqa: E402
from top_n_combination import CNN_SCORES  # noqa: E402

PROFESSOR_CSV = ROOT / "handoff/금공개 팀플/IFE_term_stock_data.csv"


def main() -> None:
    daily = Daily()
    scores = pd.read_csv(CNN_SCORES, usecols=[*KEY, "cnn_prob_up"], dtype={"formation_month": str})
    four, base = ranked(build_panel()[0].merge(scores, on=KEY, how="inner", validate="one_to_one"), True)
    four = four.assign(score=combined_score([four[c] for c in base], list(base.values())))
    professor, _ = load_professor_stock_data(PROFESSOR_CSV)
    member = official_formation_universe(professor, "2000-01", "2020-11")
    member = set(zip(member["PERMNO"].astype(int), month_code(member["formation_month"].astype(str))))
    d = daily.days[["PERMNO", "day", "l"]].reset_index(drop=True)
    level = np.exp(d.groupby("PERMNO")["l"].cumsum()).to_numpy()
    stock = {p: (g.index.to_numpy(), g.to_numpy()) for p, g in d.groupby("PERMNO")["day"]}
    month_end = pd.Series(np.arange(len(daily.day_month))).groupby(daily.day_month).max()

    def moving_averages(permno: int, end: int) -> dict:
        rows, days = stock.get(permno, (np.array([], int), np.array([], int)))
        cum = np.r_[0.0, np.cumsum(level[rows])]
        out = {}
        for n in (50, 200):
            k = np.searchsorted(days, end, side="right")  # old rule: last n valid observations
            out[f"old{n}"] = (cum[k] - cum[k - n]) / n if k >= n else np.nan
            lo, hi = np.searchsorted(days, end - n + 1), k  # team rule: last n market days, >= 80% valid
            out[f"team{n}"] = (cum[hi] - cum[lo]) / (hi - lo) if hi - lo >= int(np.ceil(0.8 * n)) else np.nan
            out[f"valid{n}"] = hi - lo
        return out

    summary, cases = [], []
    for portfolio in ["D10", "Top 50"]:
        sel = select_top(four, portfolio)
        sel = sel.loc[sel["formation_month"] <= "2019-11", KEY]
        ends = month_end.reindex(month_code(sel["formation_month"])).to_numpy()
        ma = pd.DataFrame([moving_averages(p, e) for p, e in zip(sel["PERMNO"], ends)])
        old_unknown, team_unknown = ma["old200"].isna(), ma["team50"].isna() | ma["team200"].isna()
        old_hold = ~(ma["old50"] < ma["old200"])
        team_hold = ma["team50"] > ma["team200"]
        months = sel["formation_month"].to_numpy()
        summary.append({"portfolio": portfolio, "stock_months": len(sel), "old_rule_unknown": int(old_unknown.sum()),
                        "team_rule_unknown": int(team_unknown.sum()),
                        "months_with_team_unknown": int(team_unknown.groupby(months).any().sum()),
                        "max_team_unknown_share_of_month": float(team_unknown.groupby(months).mean().max()),
                        "old_hold_team_sell": int((old_hold & ~team_hold).sum()),
                        "old_hold_team_sell_unknown": int((old_hold & team_unknown).sum()),
                        "old_sell_team_hold": int((~old_hold & team_hold).sum()),
                        "flat_ma50_equals_ma200": int((~team_unknown & (ma["team50"] == ma["team200"])).sum())})
        for (permno, fmonth), end, v50, v200 in zip(sel.loc[team_unknown.to_numpy()].itertuples(index=False),
                                                    ends[team_unknown.to_numpy()], ma.loc[team_unknown, "valid50"],
                                                    ma.loc[team_unknown, "valid200"]):
            window = np.arange(end - 199, end + 1)
            gap = ~np.isin(window, stock[permno][1])
            outside = np.array([(permno, daily.day_month[x]) not in member for x in window[gap]])
            edge = np.diff(np.r_[0, gap.astype(int), 0])
            runs = np.flatnonzero(edge == -1) - np.flatnonzero(edge == 1)
            gap_months = sorted({int(daily.day_month[x]) for x, o in zip(window[gap], outside) if o})
            cases.append({"portfolio": portfolio, "formation_month": fmonth, "valid_days_last50": int(v50),
                          "valid_days_last200": int(v200), "missing_days": int(gap.sum()),
                          "missing_outside_professor_csv": int(outside.sum()), "gap_runs": len(runs),
                          "longest_gap_days": int(runs.max()), "history_before_window": bool(stock[permno][1].min() < end - 199),
                          "_permno": permno, "_gap_months": gap_months})
    crsp = pd.read_csv(CRSP_MONTHLY, usecols=["PERMNO", "MthCalDt", "PrimaryExch", "ConditionalType", "MthRet", "MthRetFlg"])
    crsp["mcode"] = month_code(crsp["MthCalDt"].astype(str).str[:7])
    for case in cases:
        part = crsp[crsp["PERMNO"].eq(case.pop("_permno")) & crsp["mcode"].isin(months := case.pop("_gap_months"))]
        case |= {"gap_months_outside_csv": len(months), "gap_months_in_crsp": int(part["mcode"].nunique()),
                 "gap_months_with_crsp_return": int(part.loc[part["MthRet"].notna(), "mcode"].nunique()),
                 "crsp_exchange_codes": ",".join(sorted(part["PrimaryExch"].dropna().astype(str).unique())),
                 "crsp_conditional_types": ",".join(sorted(part["ConditionalType"].dropna().astype(str).unique())),
                 "crsp_return_flags": ",".join(sorted(part["MthRetFlg"].dropna().astype(str).unique()))}
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "hhz_trend_gaps_summary.csv", index=False)
    pd.DataFrame(cases).to_csv(OUTPUT_DIR / "hhz_trend_gaps_cases.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary).T.to_string())
    print(pd.DataFrame(cases).to_string(index=False))


if __name__ == "__main__":
    main()
