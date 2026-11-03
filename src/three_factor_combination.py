"""Does the original three-signal plan (Momentum + BAB + IVOL) work without CNN?

Pre-specified on 2026-09-24 before running; every design below is reported.

A. Signal combination (the team's original plan)
   - Month-t signals: momentum P(t-1)/P(t-12)-1 (``momentum_lowbeta_deciles``),
     FP beta (``fp_bab``), IVOL (professor-universe IVOL run).
   - Scores: monthly rank percentiles; momentum ascending, beta and IVOL descending
     (so a higher score is always the theoretically preferred side).
   - Combined score = equal-weight mean of the three scores. Stocks need all three signals.
   - Equal-count deciles on the combined score and, on the same sample, on each single score.
   - Formation 2004-12 to 2019-11; holding 2005-01 to 2019-12 (180 months; see ``eval_window``).
   - Samples: all stocks; formation-month CRSP |MthPrc| >= $5. Weights: EW and VW (CRSP MthCap).
   - Spread: decile 10 (best score) minus decile 1.
B. Strategy combination over the same holding months
   - IVOL EW decile spread (low minus high) and momentum EW decile spread (winners minus
     losers) on their full universes, and the FP BAB factor.
   - Equal weight (1/3 each) and inverse volatility using the trailing 36 months (t-36..t-1).
Metrics: raw mean with Newey-West(4) t, Sharpe, and CAPM / FF3 / FF5 alphas (official monthly
Kenneth French factors). Primary: design A, all stocks, EW.
All returns are delisting-corrected (``ret_final``). Only portfolio-level aggregates are written.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fp_bab  # noqa: E402
from common import assign_deciles  # noqa: E402
from delisting_robustness import load_supplement, tradable_at_formation  # noqa: E402
from eval_window import FIRST_FORMATION, LAST_FORMATION, PERIOD_TEXT  # noqa: E402
from ivol_specifications import (  # noqa: E402
    ASSIGNMENTS,
    FF3_MONTHLY_ZIP,
    FF5_MONTHLY_ZIP,
    formation_prices,
    load_french_monthly,
)
from momentum_lowbeta_deciles import load_prices, momentum_signals  # noqa: E402
from performance_metrics import newey_west_t_stat, ols_alpha_hac  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/three_factor_combination"
FIRST, LAST = FIRST_FORMATION, LAST_FORMATION
SCORES = {"MOM": "mom_score", "LOWBETA": "beta_score", "LOWIVOL": "ivol_score", "COMBINED": "combined_score"}


def pct_rank(values: pd.Series) -> pd.Series:
    return (values.rank(method="average") - 1.0) / (len(values) - 1.0)


def combined_score(ranks, weights) -> np.ndarray:
    """Weighted sum of rank columns, added in the given order and rounded to 12 decimals.

    Every combination script uses this, so the same inputs give bit-identical scores and the same ties at a
    decile boundary (a matrix product can differ in the last bit with the array layout). A weight may be a
    per-row array (e.g. a time-varying momentum weight).
    """
    total = np.zeros(len(ranks[0]))
    for rank, weight in zip(ranks, weights):
        total = total + np.asarray(weight, dtype=float) * np.asarray(rank, dtype=float)
    return np.round(total, 12)


def decile_returns(frame: pd.DataFrame, score: str, weighting: str) -> pd.DataFrame:
    data = frame.assign(decile=assign_deciles(frame, signal_column=score).astype(int)).dropna(subset=["ret_final"])
    if weighting == "VW":
        data = data.loc[data["cap"].gt(0)].assign(weighted=lambda d: d["ret_final"] * d["cap"])
        sums = data.groupby(["holding_month", "decile"])[["weighted", "cap"]].sum()
        return (sums["weighted"] / sums["cap"]).unstack("decile").sort_index()
    return data.groupby(["holding_month", "decile"])["ret_final"].mean().unstack("decile").sort_index()


def metrics(series: pd.Series, ff3: pd.DataFrame, ff5: pd.DataFrame) -> dict:
    out = {
        "months": int(len(series)),
        "raw_annual": 12 * series.mean(),
        "raw_t": newey_west_t_stat(series, lag=4),
        "sharpe": math.sqrt(12) * series.mean() / series.std(ddof=1),
        "volatility": math.sqrt(12) * series.std(ddof=1),
    }
    for name, factors, columns in [
        ("capm", ff3, ["mkt_rf"]),
        ("ff3", ff3, ["mkt_rf", "smb", "hml"]),
        ("ff5", ff5, ["mkt_rf", "smb", "hml", "rmw", "cma"]),
    ]:
        x = factors.reindex(series.index)[columns]
        fit = ols_alpha_hac(series.reset_index(drop=True), x.reset_index(drop=True), lag=4)
        out[f"{name}_alpha"] = fit["annualized_alpha"]
        out[f"{name}_t"] = fit["alpha_t_stat_newey_west"]
    return out


def build_panel(first: str = FIRST, last: str = LAST, incomplete_delisting_return: float | None = None) -> pd.DataFrame:
    """Stocks with all three signals and tradable at formation, formation months first..last.

    The 2020 stress test passes a later end; ``incomplete_delisting_return`` is the G/M/V stress (see ``load_supplement``).
    """
    ivol = pd.read_csv(
        ASSIGNMENTS,
        usecols=["PERMNO", "formation_month", "ivol_daily", "formation_market_cap"],
        dtype={"formation_month": str},
    ).rename(columns={"formation_market_cap": "cap"})
    prices, universe = load_prices()
    momentum = momentum_signals(prices, universe)["MOM"].rename(columns={"signal": "mom"})
    market = fp_bab.load_market()
    days = fp_bab.load_stock_days(market)
    stats, market_month = fp_bab.monthly_stats(days, market)
    del days
    betas = fp_bab.estimate_betas(stats, market_month)
    supplement = load_supplement(["PERMNO", "formation_month", "holding_month", "ret_final"], incomplete_delisting_return)

    panel = (
        ivol.merge(momentum, on=["PERMNO", "formation_month"], how="inner", validate="one_to_one")
        .merge(betas, on=["PERMNO", "formation_month"], how="inner", validate="one_to_one")
        .merge(supplement, on=["PERMNO", "formation_month"], how="left", validate="one_to_one")
    )
    panel = tradable_at_formation(panel.loc[panel["formation_month"].between(first, last)])
    panel = formation_prices(panel)
    return panel, ivol, momentum, supplement


def add_scores(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("formation_month")
    frame = frame.copy()
    frame["mom_score"] = grouped["mom"].transform(pct_rank)
    frame["beta_score"] = 1.0 - grouped["beta"].transform(pct_rank)
    frame["ivol_score"] = 1.0 - grouped["ivol_daily"].transform(pct_rank)
    frame["combined_score"] = combined_score([frame["mom_score"], frame["beta_score"], frame["ivol_score"]], [1 / 3] * 3)
    return frame


def main() -> None:
    ff3, ff5 = load_french_monthly(FF3_MONTHLY_ZIP), load_french_monthly(FF5_MONTHLY_ZIP)
    panel, ivol, momentum, supplement = build_panel()

    # A. signal combination
    rows, decile_rows, spreads = [], [], {}
    for sample_key in ["all", "price5"]:
        sample = panel.loc[panel["price"].ge(5.0)] if sample_key == "price5" else panel
        sample = add_scores(sample)
        for weighting in ["EW", "VW"]:
            for label, score in SCORES.items():
                wide = decile_returns(sample, score, weighting)
                spread = (wide[10] - wide[1]).dropna()
                spreads[(sample_key, weighting, label)] = spread
                rows.append({"design": "A", "sample": sample_key, "weighting": weighting, "portfolio": label,
                             "stocks_per_month": sample.groupby("formation_month").size().mean(),
                             **metrics(spread, ff3, ff5)})
                for decile in range(1, 11):
                    decile_rows.append({"sample": sample_key, "weighting": weighting, "portfolio": label,
                                        "decile": decile, "annual_mean": 12 * wide[decile].mean(),
                                        "annual_volatility": math.sqrt(12) * wide[decile].std(ddof=1)})

    # B. strategy combination on full universes
    full_ivol = ivol.merge(supplement, on=["PERMNO", "formation_month"], how="left")
    full_ivol = full_ivol.loc[full_ivol["formation_month"].between(FIRST, LAST)].assign(
        ivol_score=lambda d: -d["ivol_daily"]
    )
    full_mom = momentum.merge(supplement, on=["PERMNO", "formation_month"], how="left")
    full_mom = full_mom.loc[full_mom["formation_month"].between(FIRST, LAST)].assign(cap=np.nan)
    ivol_wide = decile_returns(full_ivol, "ivol_score", "EW")
    mom_wide = decile_returns(full_mom, "mom", "EW")
    bab = pd.read_csv(ROOT / "outputs/fp_bab/fp_bab_monthly.csv", dtype={"holding_month": str}).set_index(
        "holding_month"
    )["bab_with_delisting"]
    strategies = pd.DataFrame(
        {"IVOL": ivol_wide[10] - ivol_wide[1], "MOM": mom_wide[10] - mom_wide[1], "BAB": bab}
    ).dropna()
    equal = strategies.mean(axis=1)
    trailing_vol = strategies.rolling(36).std().shift(1)
    inverse = (1.0 / trailing_vol).div((1.0 / trailing_vol).sum(axis=1), axis=0)
    inverse_vol = (strategies * inverse).sum(axis=1, min_count=3).dropna()
    for label, series in [("IVOL", strategies["IVOL"]), ("MOM", strategies["MOM"]), ("BAB", strategies["BAB"]),
                          ("EQUAL_1/3", equal), ("INVERSE_VOL_36M", inverse_vol)]:
        rows.append({"design": "B", "sample": "full", "weighting": "EW legs", "portfolio": label,
                     **metrics(series, ff3, ff5)})

    # spanning: each strategy's alpha on market + the other two strategies
    spanning = []
    for target in strategies:
        others = [c for c in strategies if c != target]
        x = pd.concat([ff3.reindex(strategies.index)[["mkt_rf"]], strategies[others]], axis=1)
        fit = ols_alpha_hac(strategies[target].reset_index(drop=True), x.reset_index(drop=True), lag=4)
        spanning.append({"target": target, "controls": "Mkt-RF + " + " + ".join(others),
                         "alpha": fit["annualized_alpha"], "alpha_t": fit["alpha_t_stat_newey_west"]})

    result = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "combination_results.csv", index=False)
    pd.DataFrame(decile_rows).to_csv(OUTPUT_DIR / "combination_deciles.csv", index=False)
    strategies.corr().to_csv(OUTPUT_DIR / "strategy_correlations.csv")
    pd.DataFrame(spanning).to_csv(OUTPUT_DIR / "spanning_regressions.csv", index=False)

    pd.set_option("display.width", 250)
    show = ["design", "sample", "weighting", "portfolio", "raw_annual", "raw_t", "sharpe", "capm_alpha", "capm_t",
            "ff3_alpha", "ff3_t", "ff5_alpha", "ff5_t"]
    print(result[show].round(3).to_string(index=False))
    print(strategies.corr().round(3).to_string())
    print(pd.DataFrame(spanning).round(3).to_string(index=False))
    print("stocks per month (A, all):", round(result.loc[(result["design"] == "A") & (result["sample"] == "all"), "stocks_per_month"].iloc[0], 1),
          " (A, price5):", round(result.loc[(result["design"] == "A") & (result["sample"] == "price5"), "stocks_per_month"].iloc[0], 1))

    draw(pd.DataFrame(decile_rows))


def draw(deciles: pd.DataFrame) -> None:
    data = deciles.loc[(deciles["sample"] == "all") & (deciles["weighting"] == "EW")]
    styles = {
        "COMBINED": ("Combined (1/3 each)", "#D55E00", "-", "D", 2.6),
        "MOM": ("Momentum", "#0072B2", "--", "o", 1.6),
        "LOWBETA": ("Low beta (FP beta)", "#009E73", "--", "s", 1.6),
        "LOWIVOL": ("Low IVOL", "#CC79A7", "--", "^", 1.6),
    }
    with plt.rc_context({"font.family": "DejaVu Serif", "axes.edgecolor": "#6B7280", "axes.labelcolor": "#1F2937",
                         "xtick.color": "#374151", "ytick.color": "#374151", "text.color": "#111827"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharex=True)
        for axis, (metric, title, ylabel) in zip(
            axes,
            [("annual_mean", "Realized Return", "Annualized Mean Return"),
             ("annual_volatility", "Realized Volatility", "Annualized Volatility")],
        ):
            for key in ["MOM", "LOWBETA", "LOWIVOL", "COMBINED"]:
                label, color, linestyle, marker, width = styles[key]
                series = data.loc[data["portfolio"] == key].sort_values("decile")
                axis.plot(series["decile"], series[metric], color=color, linestyle=linestyle, marker=marker,
                          markersize=6, linewidth=width, label=label, zorder=3 if key == "COMBINED" else 2)
            axis.set_title(title, fontsize=12, pad=9)
            axis.set_xticks(range(1, 11))
            axis.set_xlabel("Signal Score Decile (1 = Worst, 10 = Best)")
            axis.set_ylabel(ylabel)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
            axis.set_axisbelow(True)
        axes[0].axhline(0.0, color="#4B5563", linewidth=0.9, zorder=0)
        axes[1].set_ylim(bottom=0.0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.suptitle("Momentum + BAB + IVOL Combination - Same Stocks, EW", fontsize=15, fontweight="bold", y=0.975)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=False, fontsize=9.5)
        caption = (
            f"Stocks with all three month-t signals in the professor CSV universe (EXCHCD 1/2/3); {PERIOD_TEXT}.\n"
            "Scores are monthly rank percentiles: momentum P(t-1)/P(t-12)-1 (higher is better), FP beta and IVOL (lower is better); "
            "combined = mean of the three.\n"
            "Returns are delisting-corrected (ret_final). Mean = 12 x monthly arithmetic mean; volatility = sqrt(12) x monthly sample SD."
        )
        fig.text(0.035, 0.022, caption, ha="left", va="bottom", fontsize=8.1, linespacing=1.4)
        fig.tight_layout(rect=(0.025, 0.2, 0.985, 0.875), w_pad=2.6)
        fig.savefig(OUTPUT_DIR / "combination_deciles_ew.png", dpi=200, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
