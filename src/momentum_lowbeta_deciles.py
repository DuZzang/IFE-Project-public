"""Momentum and low-beta decile sorts on the professor CSV, with and without delisting returns.

Both signals use only information available at the end of formation month t and
exact calendar months (no row-based shifts). Deciles are formed first; realized
t+1 returns are attached afterwards from ``handoff/delisting_supplement``.

- Momentum: P(t-1) / P(t-12) - 1 (skip month t). Decile 1 = losers, 10 = winners.
  The teammate notebook's P(t-2) / P(t-12) - 1 is reported as a spread-only variant.
- Low beta: CAPM beta of monthly excess returns on the Fama-French market excess
  return over months t-59..t, at least 48 valid months. Decile 1 = lowest beta.
  This is a low-beta decile sort, not the Frazzini-Pedersen beta-neutral BAB return.

Only portfolio-level aggregates are written or printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import assign_deciles, load_ff3  # noqa: E402
from delisting_robustness import OUTPUT_DIR, SUPPLEMENT, coverage, evaluate  # noqa: E402
from professor_universe import load_formation_caps  # noqa: E402

DATA_DIR = ROOT / "handoff/금공개 팀플"
PROFESSOR_CSV = DATA_DIR / "IFE_term_stock_data.csv"
CRSP_MONTHLY = DATA_DIR / "IVOL/CRSP_Monthly_1999-12_2020.csv.gz"
FF3_ZIP = DATA_DIR / "IVOL/FF3_Daily.zip"
LAST_FORMATION = "2020-11"
BETA_WINDOW = 60
BETA_MIN_OBS = 48
BETA_FIRST_FORMATION = "2004-12"  # holding months start 2005-01, as in the team's BAB notebook


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(PROFESSOR_CSV, usecols=["PERMNO", "date", "EXCHCD", "ADJ_PRC"], low_memory=False)
    frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M")
    prices = frame.pivot(index="PERMNO", columns="month", values="ADJ_PRC").sort_index(axis=1)
    full_months = pd.period_range(prices.columns.min(), prices.columns.max(), freq="M")
    prices = prices.reindex(columns=full_months)
    universe = frame.loc[frame["EXCHCD"].isin([1, 2, 3]), ["PERMNO", "month"]]
    return prices, universe


def monthly_market() -> pd.DataFrame:
    ff = load_ff3(FF3_ZIP)
    ff["month"] = ff["date"].dt.to_period("M")
    ff["mkt"] = ff["mkt_rf"] + ff["rf"]
    grouped = ff.groupby("month")
    market = pd.DataFrame(
        {
            "mkt_total": grouped["mkt"].apply(lambda s: np.prod(1.0 + s) - 1.0),
            "rf": grouped["rf"].apply(lambda s: np.prod(1.0 + s) - 1.0),
        }
    )
    market["mkt_excess"] = market["mkt_total"] - market["rf"]
    return market


def stack_signal(matrix: pd.DataFrame, name: str, universe: pd.DataFrame, first: str) -> pd.DataFrame:
    long = matrix.stack().rename(name).reset_index()
    long.columns = ["PERMNO", "month", name]
    long = long.merge(universe, on=["PERMNO", "month"], how="inner")
    long = long.loc[long["month"].between(pd.Period(first, "M"), pd.Period(LAST_FORMATION, "M"))]
    long = long.dropna(subset=[name]).copy()
    long["formation_month"] = long["month"].astype(str)
    return long[["PERMNO", "formation_month", name]]


def momentum_signals(prices: pd.DataFrame, universe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p1 = prices.shift(1, axis=1)
    p2 = prices.shift(2, axis=1)
    p12 = prices.shift(12, axis=1)
    valid = (p12 > 0)
    standard = (p1 / p12 - 1.0).where(valid & (p1 > 0))
    variant = (p2 / p12 - 1.0).where(valid & (p2 > 0))
    first = str(prices.columns.min() + 12)
    return {
        "MOM": stack_signal(standard, "signal", universe, first),
        "MOM_t2_variant": stack_signal(variant, "signal", universe, first),
    }


def beta_signal(prices: pd.DataFrame, universe: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    returns = prices / prices.shift(1, axis=1) - 1.0
    returns = returns.where((prices > 0) & (prices.shift(1, axis=1) > 0))
    market = market.reindex(returns.columns)
    excess = returns.sub(market["rf"].to_numpy(), axis=1).to_numpy()
    x_all = market["mkt_excess"].to_numpy()
    months = list(returns.columns)
    betas = np.full(excess.shape, np.nan)
    for end in range(BETA_WINDOW - 1, len(months)):
        window = slice(end - BETA_WINDOW + 1, end + 1)
        y = excess[:, window]
        x = np.broadcast_to(x_all[window], y.shape)
        valid = np.isfinite(y) & np.isfinite(x)
        n = valid.sum(axis=1)
        xv = np.where(valid, x, 0.0)
        yv = np.where(valid, y, 0.0)
        sx, sy = xv.sum(axis=1), yv.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (xv * xv).sum(axis=1) - sx * sx / n
            cov = (xv * yv).sum(axis=1) - sx * sy / n
            beta = cov / var
        betas[:, end] = np.where((n >= BETA_MIN_OBS) & (var > 1e-12), beta, np.nan)
    matrix = pd.DataFrame(betas, index=returns.index, columns=returns.columns)
    return stack_signal(matrix, "signal", universe, BETA_FIRST_FORMATION)


def attach_outcomes(signal: pd.DataFrame, supplement: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    frame = signal.copy()
    frame["decile"] = assign_deciles(frame, signal_column="signal").astype(int)
    frame = frame.merge(supplement, on=["PERMNO", "formation_month"], how="left", validate="one_to_one")
    frame = frame.merge(caps, on=["PERMNO", "formation_month"], how="left", validate="one_to_one")
    frame = frame.rename(columns={"ret_adjprc": "ret_adjprc_eval", "formation_market_cap": "cap"})
    frame["holding_month"] = (pd.PeriodIndex(frame["formation_month"], freq="M") + 1).astype(str)
    return frame


def main() -> None:
    prices, universe = load_prices()
    market = monthly_market()
    signals = momentum_signals(prices, universe)
    signals["LOWBETA"] = beta_signal(prices, universe, market)

    supplement = pd.read_csv(
        SUPPLEMENT,
        usecols=["PERMNO", "formation_month", "ret_adjprc", "ret_final", "ret_source"],
        dtype={"formation_month": str},
    )
    keys = pd.concat([s[["PERMNO", "formation_month"]] for s in signals.values()]).drop_duplicates()
    caps, _ = load_formation_caps(CRSP_MONTHLY, keys, chunksize=500_000)

    summaries, deciles, coverages, samples = [], [], [], []
    for label, (long, short) in {"MOM": (10, 1), "MOM_t2_variant": (10, 1), "LOWBETA": (1, 10)}.items():
        frame = attach_outcomes(signals[label], supplement, caps)
        summary, decile_rows = evaluate(frame, long, short, label)
        summaries += summary
        if label != "MOM_t2_variant":
            deciles += decile_rows
            coverages += coverage(frame, label)
        per_month = frame.groupby("formation_month").size()
        samples.append(
            {
                "strategy": label,
                "formation_start": frame["formation_month"].min(),
                "formation_end": frame["formation_month"].max(),
                "formation_months": int(per_month.size),
                "stock_months": int(len(frame)),
                "average_stocks_per_month": float(per_month.mean()),
                "rows_missing_cap": int(frame["cap"].isna().sum()),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "mom_lowbeta_spread_summary.csv", index=False)
    pd.DataFrame(deciles).to_csv(OUTPUT_DIR / "mom_lowbeta_decile_summary.csv", index=False)
    pd.DataFrame(coverages).to_csv(OUTPUT_DIR / "mom_lowbeta_coverage_by_decile.csv", index=False)
    pd.DataFrame(samples).to_csv(OUTPUT_DIR / "mom_lowbeta_samples.csv", index=False)

    print(pd.DataFrame(samples).to_string(index=False))
    print(pd.DataFrame(summaries).round(4).to_string(index=False))
    print(pd.DataFrame(coverages).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
