"""Frazzini-Pedersen (2014) betting-against-beta on the professor CSV universe.

Beta (FP section 3.1):
  beta_ts = rho * sigma_i / sigma_m
  - sigma: std of daily log returns over the past 12 months (at least 120 days)
  - rho:   correlation of overlapping 3-day log returns over the past 60 months
           (at least 750 observations), to allow for non-synchronous trading
  - market: Fama-French market (Mkt-RF + RF), i.e. the CRSP value-weighted market
  - shrink: beta = 0.6 * beta_ts + 0.4 * 1

Portfolios (FP eq. 16-17), rebalanced every month t:
  z = cross-sectional rank of beta, k = 2 / sum|z - mean z|
  w_H = k (z - mean z)+,  w_L = k (z - mean z)-   (each leg sums to 1)
  r_BAB(t+1) = (r_L - r_f) / beta_L - (r_H - r_f) / beta_H,  beta_L = w_L'beta, beta_H = w_H'beta

The t+1 return is either the professor ADJ_PRC return or ``ret_final`` from
``handoff/delisting_supplement``. A stock without a t+1 return is dropped from its
leg and the leg's remaining weights are rescaled; the leg beta stays ex ante.

Inputs are the daily panel built for IVOL (professor-universe stock-days, CRSP DlyRet
with FF3) and the FF3 daily file. Only portfolio-level monthly series are written.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "handoff/ivol_combination/src"))
from common import load_ff3  # noqa: E402
from eval_window import END as EVAL_END, START as EVAL_START  # noqa: E402
from performance_metrics import maximum_drawdown, newey_west_t_stat, ols_alpha_hac  # noqa: E402

DAILY_PANEL = ROOT / "derived/professor_universe_2000_2020/daily_panel_2000-01_2020-11.csv.gz"
FF3_ZIP = ROOT / "handoff/금공개 팀플/IVOL/FF3_Daily.zip"
SUPPLEMENT = ROOT / "handoff/delisting_supplement/next_month_returns_with_delisting_2000-2020.csv.gz"
OUTPUT_DIR = ROOT / "outputs/fp_bab"

VOL_MONTHS, MIN_VOL_DAYS = 12, 120
CORR_MONTHS, MIN_CORR_OBS = 60, 750
SHRINK_WEIGHT = 0.6
FIRST_FORMATION, LAST_FORMATION = "2004-12", "2020-11"
PANEL_START = pd.Period("2000-01", "M")


def load_market() -> pd.DataFrame:
    ff = load_ff3(FF3_ZIP)
    ff = ff.loc[ff["date"].between("1999-12-01", "2020-12-31")].reset_index(drop=True)
    ff["lm"] = np.log1p(ff["mkt_rf"] + ff["rf"])
    ff["lm3"] = ff["lm"].rolling(3).sum()
    ff["month"] = ff["date"].dt.to_period("M")
    ff["key"] = ff["date"].dt.strftime("%Y-%m-%d")
    return ff


def load_stock_days(market: pd.DataFrame) -> pd.DataFrame:
    day_index = pd.Index(market["key"])
    pieces = []
    for chunk in pd.read_csv(
        DAILY_PANEL,
        usecols=["PERMNO", "date", "excess_return", "rf"],
        dtype={"date": str},
        chunksize=3_000_000,
    ):
        day = day_index.get_indexer(chunk["date"].str[:10])
        if (day < 0).any():
            raise ValueError("Daily panel has dates missing from the FF3 calendar.")
        log_return = np.log1p(chunk["excess_return"].to_numpy() + chunk["rf"].to_numpy())
        pieces.append(
            pd.DataFrame(
                {"PERMNO": chunk["PERMNO"].to_numpy(np.int32), "day": day.astype(np.int32), "l": log_return}
            )
        )
    days = pd.concat(pieces, ignore_index=True)
    days = days.loc[np.isfinite(days["l"])]
    days = days.sort_values(["PERMNO", "day"]).drop_duplicates(["PERMNO", "day"]).reset_index(drop=True)
    return days


def monthly_stats(days: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    perm = days["PERMNO"].to_numpy()
    day = days["day"].to_numpy()
    l = days["l"].to_numpy()
    consecutive = np.zeros(len(days), dtype=bool)
    consecutive[2:] = (perm[2:] == perm[:-2]) & (day[2:] - day[:-2] == 2)
    l3 = np.full(len(days), np.nan)
    l3[2:] = np.where(consecutive[2:], l[2:] + l[1:-1] + l[:-2], np.nan)
    m3 = market["lm3"].to_numpy()[day]
    pair = np.isfinite(l3) & np.isfinite(m3)

    month_of_day = np.asarray((market["month"] - PANEL_START).apply(lambda offset: offset.n), dtype=np.int32)
    frame = pd.DataFrame(
        {
            "PERMNO": perm,
            "m": month_of_day[day],
            "n1": 1.0,
            "s1": l,
            "q1": l * l,
            "n3": pair.astype(float),
            "sx": np.where(pair, l3, 0.0),
            "sy": np.where(pair, m3, 0.0),
            "sxx": np.where(pair, l3 * l3, 0.0),
            "syy": np.where(pair, m3 * m3, 0.0),
            "sxy": np.where(pair, l3 * m3, 0.0),
        }
    )
    stats = frame.groupby(["PERMNO", "m"], sort=True).sum().reset_index()

    market_month = pd.DataFrame(
        {"m": month_of_day, "n": 1.0, "s": market["lm"].to_numpy(), "q": market["lm"].to_numpy() ** 2}
    )
    market_month = market_month.loc[market_month["m"] >= 0].groupby("m").sum()
    return stats, market_month


def rolling(grid: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(grid, axis=-1)
    result = cumulative.copy()
    result[..., window:] = cumulative[..., window:] - cumulative[..., :-window]
    return result


def estimate_betas(stats: pd.DataFrame, market_month: pd.DataFrame) -> pd.DataFrame:
    permnos = np.sort(stats["PERMNO"].unique())
    n_months = int(stats["m"].max()) + 1
    row = np.searchsorted(permnos, stats["PERMNO"].to_numpy())
    col = stats["m"].to_numpy()

    def grid(column: str) -> np.ndarray:
        values = np.zeros((len(permnos), n_months))
        values[row, col] = stats[column].to_numpy()
        return values

    n1, s1, q1 = (rolling(grid(c), VOL_MONTHS) for c in ["n1", "s1", "q1"])
    n3, sx, sy, sxx, syy, sxy = (rolling(grid(c), CORR_MONTHS) for c in ["n3", "sx", "sy", "sxx", "syy", "sxy"])

    mm = market_month.reindex(range(n_months), fill_value=0.0)
    mn, ms, mq = (rolling(mm[c].to_numpy(), VOL_MONTHS) for c in ["n", "s", "q"])
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_i = np.sqrt((q1 - s1 * s1 / n1) / (n1 - 1))
        sigma_m = np.sqrt((mq - ms * ms / mn) / (mn - 1))
        cov = sxy - sx * sy / n3
        rho = cov / np.sqrt((sxx - sx * sx / n3) * (syy - sy * sy / n3))
        beta_ts = rho * sigma_i / sigma_m[np.newaxis, :]
    beta_ts[(n1 < MIN_VOL_DAYS) | (n3 < MIN_CORR_OBS) | ~np.isfinite(beta_ts)] = np.nan
    beta = SHRINK_WEIGHT * beta_ts + (1.0 - SHRINK_WEIGHT)

    months = [str(PANEL_START + i) for i in range(n_months)]
    result = pd.DataFrame(beta, index=permnos, columns=months).stack().rename("beta").reset_index()
    result.columns = ["PERMNO", "formation_month", "beta"]
    return result.dropna(subset=["beta"])


def build_bab(betas: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    supplement = pd.read_csv(
        SUPPLEMENT,
        usecols=["PERMNO", "formation_month", "holding_month", "ret_adjprc", "ret_final"],
        dtype={"formation_month": str, "holding_month": str},
    )
    panel = supplement.merge(betas, on=["PERMNO", "formation_month"], how="inner", validate="one_to_one")
    panel = panel.loc[panel["formation_month"].between(FIRST_FORMATION, LAST_FORMATION)]

    compound = lambda s: np.prod(1.0 + s) - 1.0  # noqa: E731
    monthly_rf = market.groupby("month")["rf"].apply(compound)
    monthly_mkt = (market["mkt_rf"] + market["rf"]).groupby(market["month"]).apply(compound) - monthly_rf

    records = []
    for month, group in panel.groupby("formation_month", sort=True):
        beta = group["beta"].to_numpy()
        z = pd.Series(beta).rank(method="average").to_numpy()
        centered = z - z.mean()
        k = 2.0 / np.abs(centered).sum()
        w_high = k * np.clip(centered, 0.0, None)
        w_low = k * np.clip(-centered, 0.0, None)
        beta_high, beta_low = float(w_high @ beta), float(w_low @ beta)
        holding = pd.Period(month, "M") + 1
        rf = float(monthly_rf.loc[holding])
        record = {
            "formation_month": month,
            "holding_month": str(holding),
            "stocks": int(len(group)),
            "beta_low": beta_low,
            "beta_high": beta_high,
            "rf": rf,
            "mkt_excess": float(monthly_mkt.loc[holding]),
        }
        for version, column in [("adj_prc_only", "ret_adjprc"), ("with_delisting", "ret_final")]:
            returns = group[column].to_numpy()
            available = np.isfinite(returns)
            r_low = float((w_low[available] @ returns[available]) / w_low[available].sum())
            r_high = float((w_high[available] @ returns[available]) / w_high[available].sum())
            record[f"r_low_{version}"] = r_low
            record[f"r_high_{version}"] = r_high
            record[f"bab_{version}"] = (r_low - rf) / beta_low - (r_high - rf) / beta_high
            record[f"missing_{version}"] = int((~available).sum())
        records.append(record)
    return pd.DataFrame(records)


def summarize(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    windows = {
        f"{EVAL_START}~{EVAL_END}": (EVAL_START, EVAL_END),
        f"{EVAL_START}~2012-12": (EVAL_START, "2012-12"),
        f"2013-01~{EVAL_END}": ("2013-01", EVAL_END),
    }
    for version in ["adj_prc_only", "with_delisting"]:
        for label, (start, end) in windows.items():
            part = monthly.loc[monthly["holding_month"].between(start, end)]
            bab = part[f"bab_{version}"].reset_index(drop=True)
            mkt = part["mkt_excess"].reset_index(drop=True)
            alpha = ols_alpha_hac(bab, mkt.to_frame("mkt"), lag=4)
            slope = float(np.polyfit(mkt, bab, 1)[0])
            rows.append(
                {
                    "version": version,
                    "period": label,
                    "months": int(len(bab)),
                    "annualized_mean": 12 * bab.mean(),
                    "annualized_volatility": math.sqrt(12) * bab.std(ddof=1),
                    "sharpe": math.sqrt(12) * bab.mean() / bab.std(ddof=1),
                    "newey_west_t_lag4": newey_west_t_stat(bab, lag=4),
                    "capm_alpha_annualized": alpha["annualized_alpha"],
                    "capm_alpha_t_nw4": alpha["alpha_t_stat_newey_west"],
                    "realized_market_beta": slope,
                    "maximum_drawdown": maximum_drawdown(bab),
                    "cumulative_return": float(np.prod(1.0 + bab) - 1.0),
                    "mean_beta_low": part["beta_low"].mean(),
                    "mean_beta_high": part["beta_high"].mean(),
                    "mean_stocks": part["stocks"].mean(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if "--summary-only" in sys.argv:  # re-summarize the saved monthly factor on the evaluation window
        monthly = pd.read_csv(OUTPUT_DIR / "fp_bab_monthly.csv", dtype={"holding_month": str})
        summary = summarize(monthly)
        summary.to_csv(OUTPUT_DIR / "fp_bab_summary.csv", index=False)
        print(summary.round(4).to_string(index=False))
        return
    market = load_market()
    days = load_stock_days(market)
    stats, market_month = monthly_stats(days, market)
    del days
    betas = estimate_betas(stats, market_month)
    monthly = build_bab(betas, market)
    summary = summarize(monthly)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_DIR / "fp_bab_monthly.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "fp_bab_summary.csv", index=False)
    beta_profile = betas.loc[betas["formation_month"].between(FIRST_FORMATION, LAST_FORMATION), "beta"]
    audit = {
        "stock_months_with_beta_in_formation_window": int(len(beta_profile)),
        "beta_percentiles": {str(q): float(beta_profile.quantile(q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]},
        "formation_months": int(len(monthly)),
        "missing_returns_total": {
            v: int(monthly[f"missing_{v}"].sum()) for v in ["adj_prc_only", "with_delisting"]
        },
    }
    (OUTPUT_DIR / "fp_bab_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
