"""Pre-specified IVOL robustness set (declared 2026-09-24, before any of these were run).

All designs use the delisting-corrected next-month return (``ret_final``), the same
month-t IVOL signal, and holding months 2000-02 to 2020-12. Every design is reported;
none is dropped after seeing results.

Designs (low-IVOL minus high-IVOL spread):
  1 EW deciles, all stocks (project baseline)
  2 VW deciles, all stocks
  3 EW quintiles
  4 VW quintiles (Ang, Hodrick, Xing and Zhang 2006 main design)
  5 EW deciles, formation-month price >= $5
  6 VW deciles, formation-month price >= $5
  7 EW deciles, NYSE breakpoints (Bali and Cakici 2008 check)
  8 VW deciles, NYSE breakpoints
Metrics for each: raw mean spread, CAPM alpha, Fama-French 3-factor alpha, all with
Newey-West(4) t-statistics. The FF3 alpha is the primary metric (as in Ang et al.).

Factor data: the official Kenneth French monthly FF3 file is used for CAPM and FF3
alphas. The first run (2026-09-24) used FF3 factors compounded from the daily file;
those approximate alphas are kept in ``ff3_approx_*`` columns for comparison.

Added after the first run, disclosed here: FF5 alphas (Mkt-RF, SMB, HML, RMW, CMA) and the
calendar-2000 exclusion. Both are checks that can only weaken the result.
Only portfolio-level aggregates are written or printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "handoff/ivol_combination/src"))
from common import load_ff3  # noqa: E402
from performance_metrics import newey_west_t_stat, ols_alpha_hac  # noqa: E402

ASSIGNMENTS = ROOT / "derived/professor_universe_2000_2020/ivol_deciles_adj_prc_returns_2000-2020.csv.gz"
SUPPLEMENT = ROOT / "handoff/delisting_supplement/next_month_returns_with_delisting_2000-2020.csv.gz"
CRSP_MONTHLY = ROOT / "handoff/금공개 팀플/IVOL/CRSP_Monthly_1999-12_2020.csv.gz"
FACTOR_DIR = ROOT / "handoff/금공개 팀플/IVOL"
FF3_ZIP = FACTOR_DIR / "FF3_Daily.zip"
FF3_MONTHLY_ZIP = FACTOR_DIR / "F-F_Research_Data_Factors_CSV.zip"
FF5_MONTHLY_ZIP = FACTOR_DIR / "F-F_Research_Data_5_Factors_2x3_CSV.zip"
OUTPUT_DIR = ROOT / "outputs/ivol_specifications"

DESIGNS = [
    # (id, label, weighting, groups, filter, breakpoints)
    (1, "EW deciles (baseline)", "EW", 10, None, "all"),
    (2, "VW deciles", "VW", 10, None, "all"),
    (3, "EW quintiles", "EW", 5, None, "all"),
    (4, "VW quintiles (Ang et al.)", "VW", 5, None, "all"),
    (5, "EW deciles, price >= $5", "EW", 10, "price5", "all"),
    (6, "VW deciles, price >= $5", "VW", 10, "price5", "all"),
    (7, "EW deciles, NYSE breakpoints", "EW", 10, None, "nyse"),
    (8, "VW deciles, NYSE breakpoints", "VW", 10, None, "nyse"),
]


def load_french_monthly(zip_path: Path) -> pd.DataFrame:
    """Monthly rows (YYYYMM) of a Kenneth French CSV zip, in decimals, indexed by 'YYYY-MM'."""
    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8-sig", errors="replace")
    header, rows = None, []
    for line in text.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if header is None and len(fields) > 1 and fields[0] == "" and fields[1]:
            header = [name.lower().replace("-", "_") for name in fields[1:]]
            continue
        if header and len(fields[0]) == 6 and fields[0].isdigit():
            rows.append([f"{fields[0][:4]}-{fields[0][4:]}", *map(float, fields[1 : len(header) + 1])])
    frame = pd.DataFrame(rows, columns=["month", *header]).drop_duplicates("month").set_index("month")
    return frame / 100.0


def monthly_factors() -> pd.DataFrame:
    """FF3 compounded from the daily file (the first run's approximation, kept for comparison)."""
    ff = load_ff3(FF3_ZIP)
    ff["month"] = ff["date"].dt.to_period("M").astype(str)
    compound = lambda s: np.prod(1.0 + s) - 1.0  # noqa: E731
    grouped = ff.assign(mkt=ff["mkt_rf"] + ff["rf"]).groupby("month")
    factors = pd.DataFrame(
        {
            "mkt": grouped["mkt"].apply(compound),
            "rf": grouped["rf"].apply(compound),
            "smb": grouped["smb"].apply(compound),
            "hml": grouped["hml"].apply(compound),
        }
    )
    factors["mkt_rf"] = factors["mkt"] - factors["rf"]
    return factors[["mkt_rf", "smb", "hml", "rf"]]


def formation_prices(keys: pd.DataFrame) -> pd.DataFrame:
    crsp = pd.read_csv(
        CRSP_MONTHLY, usecols=["PERMNO", "MthCalDt", "MthPrc"], keep_default_na=False, na_values=[""], low_memory=False
    )
    crsp["formation_month"] = pd.to_datetime(crsp["MthCalDt"], errors="coerce").dt.to_period("M").astype(str)
    crsp["price"] = pd.to_numeric(crsp["MthPrc"], errors="coerce").abs()
    crsp = crsp.dropna(subset=["price"]).drop_duplicates(["PERMNO", "formation_month"], keep="first")
    return keys.merge(crsp[["PERMNO", "formation_month", "price"]], on=["PERMNO", "formation_month"], how="left")


def assign_groups(frame: pd.DataFrame, groups: int, breakpoints: str) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for _, month in frame.groupby("formation_month", sort=True):
        if breakpoints == "nyse":
            nyse = month.loc[month["EXCHCD"] == 1, "ivol_daily"]
            cuts = np.quantile(nyse, np.arange(1, groups) / groups)
            result.loc[month.index] = np.searchsorted(cuts, month["ivol_daily"].to_numpy(), side="right") + 1
        else:
            ordered = month.sort_values(["ivol_daily", "PERMNO"], kind="mergesort")
            bins = np.arange(len(ordered)) * groups // len(ordered) + 1
            result.loc[ordered.index] = bins
    return result.astype(int)


def spread_series(frame: pd.DataFrame, weighting: str, groups: int) -> tuple[pd.Series, dict]:
    usable = frame.dropna(subset=["ret_final"])
    if weighting == "VW":
        usable = usable.loc[usable["formation_market_cap"].gt(0)].assign(
            weighted=lambda d: d["ret_final"] * d["formation_market_cap"]
        )
        sums = usable.groupby(["holding_month", "group"])[["weighted", "formation_market_cap"]].sum()
        returns = (sums["weighted"] / sums["formation_market_cap"]).unstack("group")
    else:
        returns = usable.groupby(["holding_month", "group"])["ret_final"].mean().unstack("group")
    counts = usable.groupby(["holding_month", "group"]).size().unstack("group")
    spread = (returns[1] - returns[groups]).dropna()
    info = {
        "low_group_annual": 12 * returns[1].mean(),
        "high_group_annual": 12 * returns[groups].mean(),
        "avg_stocks_low": counts[1].mean(),
        "avg_stocks_high": counts[groups].mean(),
    }
    return spread, info


def main() -> None:
    frame = pd.read_csv(
        ASSIGNMENTS,
        usecols=["PERMNO", "formation_month", "EXCHCD", "ivol_daily", "formation_market_cap"],
        dtype={"formation_month": str},
    )
    supplement = pd.read_csv(
        SUPPLEMENT, usecols=["PERMNO", "formation_month", "holding_month", "ret_final"], dtype={"formation_month": str, "holding_month": str}
    )
    frame = frame.merge(supplement, on=["PERMNO", "formation_month"], how="left", validate="one_to_one")
    frame = formation_prices(frame)
    approx = monthly_factors()
    ff3 = load_french_monthly(FF3_MONTHLY_ZIP)
    ff5 = load_french_monthly(FF5_MONTHLY_ZIP)
    ff3_cols, ff5_cols = ["mkt_rf", "smb", "hml"], ["mkt_rf", "smb", "hml", "rmw", "cma"]

    def alpha(spread: pd.Series, factors: pd.DataFrame, columns: list[str]) -> dict:
        x = factors.reindex(spread.index)[columns]
        if x.isna().any().any():
            raise ValueError(f"Factor months missing for {columns}")
        return ols_alpha_hac(spread.reset_index(drop=True), x.reset_index(drop=True), lag=4)

    def loadings(spread: pd.Series, factors: pd.DataFrame, columns: list[str]) -> np.ndarray:
        x = np.column_stack([np.ones(len(spread)), factors.reindex(spread.index)[columns].to_numpy()])
        return np.linalg.lstsq(x, spread.to_numpy(), rcond=None)[0][1:]

    rows = []
    for design_id, label, weighting, groups, stock_filter, breakpoints in DESIGNS:
        sample = frame.loc[frame["price"].ge(5.0)] if stock_filter == "price5" else frame
        sample = sample.assign(group=assign_groups(sample, groups, breakpoints))
        spread, info = spread_series(sample, weighting, groups)  # zero-investment long-short
        later = spread[spread.index >= "2001-01"]
        capm, a3, a5 = alpha(spread, ff3, ["mkt_rf"]), alpha(spread, ff3, ff3_cols), alpha(spread, ff5, ff5_cols)
        a3_ex, a5_ex = alpha(later, ff3, ff3_cols), alpha(later, ff5, ff5_cols)
        a3_approx = alpha(spread, approx, ff3_cols)
        b5 = loadings(spread, ff5, ff5_cols)
        rows.append(
            {
                "design": design_id,
                "label": label,
                "months": int(len(spread)),
                "raw_annual": 12 * spread.mean(),
                "raw_t": newey_west_t_stat(spread, lag=4),
                "capm_alpha_annual": capm["annualized_alpha"],
                "capm_t": capm["alpha_t_stat_newey_west"],
                "ff3_alpha_annual": a3["annualized_alpha"],
                "ff3_t": a3["alpha_t_stat_newey_west"],
                "ff5_alpha_annual": a5["annualized_alpha"],
                "ff5_t": a5["alpha_t_stat_newey_west"],
                "ff3_alpha_ex2000": a3_ex["annualized_alpha"],
                "ff3_t_ex2000": a3_ex["alpha_t_stat_newey_west"],
                "ff5_alpha_ex2000": a5_ex["annualized_alpha"],
                "ff5_t_ex2000": a5_ex["alpha_t_stat_newey_west"],
                "ff3_approx_alpha_annual": a3_approx["annualized_alpha"],
                "ff3_approx_t": a3_approx["alpha_t_stat_newey_west"],
                **{f"ff3_beta_{c}": b for c, b in zip(ff3_cols, loadings(spread, ff3, ff3_cols))},
                **{f"ff5_beta_{c}": b for c, b in zip(ff5_cols, b5)},
                **info,
            }
        )

    months = [str(m) for m in pd.period_range("2000-02", "2020-12", freq="M")]
    approximation = {
        column: {
            "correlation": float(approx.loc[months, column].corr(ff3.loc[months, column])),
            "max_abs_difference": float((approx.loc[months, column] - ff3.loc[months, column]).abs().max()),
            "mean_abs_difference": float((approx.loc[months, column] - ff3.loc[months, column]).abs().mean()),
        }
        for column in ff3_cols
    }

    result = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "ivol_specification_results.csv", index=False)
    pd.Series(approximation).to_json(OUTPUT_DIR / "ff3_daily_approximation_check.json", indent=2)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(approximation).round(6).to_string())
    print(
        result[
            ["design", "label", "raw_annual", "raw_t", "capm_alpha_annual", "capm_t", "ff3_alpha_annual", "ff3_t",
             "ff5_alpha_annual", "ff5_t", "ff3_t_ex2000", "ff5_alpha_ex2000", "ff5_t_ex2000", "ff3_approx_t"]
        ].round(4).to_string(index=False)
    )
    print(result[["design", *[c for c in result.columns if c.startswith("ff5_beta_")]]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
