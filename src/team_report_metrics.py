"""Unified performance table, turnover and trading-cost analysis for the team report (2026-09-24).

Uses the same definitions everywhere:
  annual mean = 12 x monthly mean; volatility = sqrt(12) x monthly SD; t = Newey-West(4);
  long-short spreads are zero-investment (Sharpe = mean / SD x sqrt(12), alphas on the raw
  spread); long-only portfolios use excess returns over RF; MDD starts from a value of 1.
Name turnover = share of a decile's (or top-N's) stocks that are new versus the previous month.
Only portfolio-level aggregates are written.
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
from common import assign_deciles  # noqa: E402
from delisting_robustness import CNN, IVOL, SUPPLEMENT, monthly_portfolios  # noqa: E402
from eval_window import END, FIRST_FORMATION, LAST_FORMATION, START, clip  # noqa: E402
from ivol_specifications import FF3_MONTHLY_ZIP, FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402
from momentum_lowbeta_deciles import (  # noqa: E402
    CRSP_MONTHLY,
    attach_outcomes,
    beta_signal,
    load_prices,
    momentum_signals,
    monthly_market,
)
from performance_metrics import maximum_drawdown, newey_west_t_stat, ols_alpha_hac  # noqa: E402
from professor_universe import load_formation_caps  # noqa: E402
from three_factor_combination import add_scores, build_panel  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/team_report"
FF5_COLS = ["mkt_rf", "smb", "hml", "rmw", "cma"]


def perf(series: pd.Series, ff3: pd.DataFrame, ff5: pd.DataFrame, long_only: bool) -> dict:
    series = series.dropna()
    series.index = series.index.astype(str)
    excess = series - ff5["rf"].reindex(series.index) if long_only else series
    out = {
        "start": series.index.min(),
        "end": series.index.max(),
        "months": int(len(series)),
        "annual_mean": 12 * series.mean(),
        "volatility": math.sqrt(12) * series.std(ddof=1),
        "sharpe": math.sqrt(12) * excess.mean() / excess.std(ddof=1),
        "nw_t": newey_west_t_stat(excess, lag=4),
        "max_drawdown": maximum_drawdown(series),
        "cagr": float(np.prod(1.0 + series)) ** (12 / len(series)) - 1,
    }
    for name, factors, columns in [("capm", ff3, ["mkt_rf"]), ("ff3", ff3, ["mkt_rf", "smb", "hml"]), ("ff5", ff5, FF5_COLS)]:
        x = factors.reindex(series.index)[columns]
        fit = ols_alpha_hac(excess.reset_index(drop=True), x.reset_index(drop=True), lag=4)
        out[f"{name}_alpha"] = fit["annualized_alpha"]
        out[f"{name}_t"] = fit["alpha_t_stat_newey_west"]
    return out


def name_turnover(frame: pd.DataFrame, decile: int) -> float:
    members = frame.loc[frame["decile"] == decile].groupby("formation_month")["PERMNO"].apply(set).sort_index()
    values, previous = [], None
    for current in members:
        if previous is not None and current:
            values.append(1.0 - len(current & previous) / len(current))
        previous = current
    return float(np.mean(values))


def main() -> None:
    ff3, ff5 = load_french_monthly(FF3_MONTHLY_ZIP), load_french_monthly(FF5_MONTHLY_ZIP)
    supplement = pd.read_csv(
        SUPPLEMENT, usecols=["PERMNO", "formation_month", "ret_adjprc", "ret_final", "ret_source"],
        dtype={"formation_month": str},
    )
    rows, turnover = [], []

    def add_spreads(label: str, frame: pd.DataFrame, long: int, short: int) -> None:
        for version, column in [("before", "ret_adjprc_eval"), ("after", "ret_final")]:
            for weighting, weighted in [("EW", False), ("VW", True)]:
                wide = clip(monthly_portfolios(frame, column, weighted))
                rows.append({"strategy": label, "version": version, "weighting": weighting, "portfolio": f"Q{long}-Q{short}",
                             **perf(wide[long] - wide[short], ff3, ff5, long_only=False)})
        formed = frame.loc[frame["formation_month"].between(FIRST_FORMATION, LAST_FORMATION)]
        turnover.append({"strategy": label, "long_decile": long, "long_leg_name_turnover": name_turnover(formed, long),
                         "short_leg_name_turnover": name_turnover(formed, short)})

    # IVOL (decile 1 = low IVOL)
    ivol = pd.read_csv(
        IVOL, usecols=["PERMNO", "formation_month", "decile", "next_month", "next_month_return", "formation_market_cap"],
        dtype={"formation_month": str, "next_month": str},
    ).rename(columns={"next_month": "holding_month", "next_month_return": "ret_adjprc_eval", "formation_market_cap": "cap"})
    ivol = ivol.merge(supplement[["PERMNO", "formation_month", "ret_final"]], on=["PERMNO", "formation_month"], how="left")
    add_spreads("IVOL", ivol, 1, 10)

    # CNN (decile 10 = highest P(up))
    cnn = pd.read_csv(
        CNN, usecols=["PERMNO", "formation_month", "holding_month", "cnn_decile", "next_month_return", "formation_market_cap"],
        dtype={"formation_month": str, "holding_month": str},
    ).rename(columns={"cnn_decile": "decile", "next_month_return": "ret_adjprc_eval", "formation_market_cap": "cap"})
    cnn = cnn.merge(supplement[["PERMNO", "formation_month", "ret_final"]], on=["PERMNO", "formation_month"], how="left")
    add_spreads("CNN", cnn, 10, 1)

    # Momentum and monthly-beta deciles
    prices, universe = load_prices()
    signals = momentum_signals(prices, universe)
    signals["LOWBETA"] = beta_signal(prices, universe, monthly_market())
    keys = pd.concat([signals[k][["PERMNO", "formation_month"]] for k in ["MOM", "LOWBETA"]]).drop_duplicates()
    caps, _ = load_formation_caps(CRSP_MONTHLY, keys, chunksize=500_000)
    add_spreads("MOM", attach_outcomes(signals["MOM"], supplement, caps), 10, 1)
    add_spreads("BETA_DECILES", attach_outcomes(signals["LOWBETA"], supplement, caps), 1, 10)

    # FP BAB
    bab = pd.read_csv(ROOT / "outputs/fp_bab/fp_bab_monthly.csv", dtype={"holding_month": str}).set_index("holding_month")
    for version, column in [("before", "bab_adj_prc_only"), ("after", "bab_with_delisting")]:
        rows.append({"strategy": "FP_BAB", "version": version, "weighting": "rank", "portfolio": "BAB",
                     **perf(clip(bab[column]), ff3, ff5, long_only=False)})

    # Combination deciles (3 signals, after correction)
    panel, *_ = build_panel()
    scored = add_scores(panel)
    scored = scored.assign(decile=assign_deciles(scored, signal_column="combined_score").astype(int), ret_adjprc_eval=np.nan)
    for weighting, weighted in [("EW", False), ("VW", True)]:
        wide = monthly_portfolios(scored, "ret_final", weighted)
        rows.append({"strategy": "COMBINED_3", "version": "after", "weighting": weighting, "portfolio": "Q10-Q1",
                     **perf(wide[10] - wide[1], ff3, ff5, long_only=False)})
        rows.append({"strategy": "COMBINED_3", "version": "after", "weighting": weighting, "portfolio": "Q10 long-only",
                     **perf(wide[10], ff3, ff5, long_only=True)})
    turnover.append({"strategy": "COMBINED_3", "long_decile": 10, "long_leg_name_turnover": name_turnover(scored, 10),
                     "short_leg_name_turnover": name_turnover(scored, 1)})

    # Top-50 portfolios and benchmarks
    top = pd.read_csv(ROOT / "outputs/top_n_combination/top_n_monthly.csv", header=[0, 1], index_col=0)
    top.index = top.index.astype(str)
    top = clip(top)
    bench = pd.read_csv(ROOT / "outputs/top_n_combination/benchmarks_monthly.csv", index_col=0)
    bench.index = bench.index.astype(str)
    bench = clip(bench)
    for key in ["P  top50 EW all", "C4 top50 EW 4 signals"]:
        for kind in ["gross", "net"]:
            rows.append({"strategy": key.strip(), "version": kind, "weighting": "EW", "portfolio": "top 50",
                         **perf(top[(key, kind)].astype(float), ff3, ff5, long_only=True)})
        turnover.append({"strategy": key.strip(), "long_decile": np.nan,
                         "long_leg_name_turnover": top[(key, "traded")].astype(float).iloc[1:].mean() / 2,
                         "short_leg_name_turnover": np.nan})
    for key in ["Market (FF Mkt)", "Universe EW"]:
        rows.append({"strategy": key, "version": "benchmark", "weighting": "", "portfolio": "",
                     **perf(bench[key].astype(float), ff3, ff5, long_only=True)})

    # Trading-cost grid for the top-50 portfolios
    grid_rows = []
    market = bench["Market (FF Mkt)"].astype(float)
    for key in ["P  top50 EW all", "C4 top50 EW 4 signals"]:
        gross, traded = top[(key, "gross")].astype(float).dropna(), top[(key, "traded")].astype(float).dropna()
        market_sharpe = perf(market.reindex(gross.index), ff3, ff5, long_only=True)["sharpe"]
        for cost in np.round(np.arange(0.0, 0.00501, 0.0001), 6):
            p = perf(gross - cost * traded, ff3, ff5, long_only=True)
            grid_rows.append({"strategy": key.strip(), "cost_per_dollar": cost, "annual_mean": p["annual_mean"],
                              "sharpe": p["sharpe"], "capm_alpha": p["capm_alpha"], "capm_t": p["capm_t"],
                              "ff5_alpha": p["ff5_alpha"], "ff5_t": p["ff5_t"], "market_sharpe": market_sharpe})
    grid = pd.DataFrame(grid_rows)

    def break_even(frame: pd.DataFrame, condition) -> float:
        failing = frame.loc[condition(frame), "cost_per_dollar"]
        return float(failing.min()) if len(failing) else np.nan

    breakeven = []
    for key, frame in grid.groupby("strategy"):
        breakeven.append({
            "strategy": key,
            "cost_where_capm_alpha_t_below_1.96": break_even(frame, lambda d: d["capm_t"] < 1.96),
            "cost_where_capm_alpha_below_0": break_even(frame, lambda d: d["capm_alpha"] <= 0),
            "cost_where_sharpe_below_market": break_even(frame, lambda d: d["sharpe"] <= d["market_sharpe"]),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False)
    pd.DataFrame(turnover).to_csv(OUTPUT_DIR / "turnover.csv", index=False)
    grid.to_csv(OUTPUT_DIR / "top50_cost_grid.csv", index=False)
    pd.DataFrame(breakeven).to_csv(OUTPUT_DIR / "top50_breakeven.csv", index=False)

    pd.set_option("display.width", 260)
    show = ["strategy", "version", "weighting", "portfolio", "start", "end", "months", "annual_mean", "volatility", "sharpe",
            "nw_t", "max_drawdown", "capm_alpha", "capm_t", "ff3_alpha", "ff3_t", "ff5_alpha", "ff5_t"]
    print(result[show].round(3).to_string(index=False))
    print(pd.DataFrame(turnover).round(3).to_string(index=False))
    print(pd.DataFrame(breakeven).to_string(index=False))
    print(grid.loc[grid["cost_per_dollar"].isin([0.0, 0.0005, 0.001, 0.002, 0.003, 0.005])].round(4).to_string(index=False))
    draw_costs(grid)


def draw_costs(grid: pd.DataFrame) -> None:
    styles = {"P  top50 EW all": ("Top 50, 3 signals", "#D55E00", "-"), "C4 top50 EW 4 signals": ("Top 50, 4 signals (with CNN)", "#CC79A7", "--")}
    with plt.rc_context({"font.family": "DejaVu Serif", "axes.edgecolor": "#6B7280", "axes.labelcolor": "#1F2937",
                         "xtick.color": "#374151", "ytick.color": "#374151", "text.color": "#111827"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
        for key, (label, color, style) in styles.items():
            data = grid.loc[grid["strategy"] == key.strip()]
            axes[0].plot(data["cost_per_dollar"], data["sharpe"], color=color, linestyle=style, linewidth=2.2, label=label)
            axes[1].plot(data["cost_per_dollar"], data["capm_t"], color=color, linestyle=style, linewidth=2.2, label=label)
        market = grid.loc[grid["strategy"] == "P  top50 EW all".strip(), "market_sharpe"].iloc[0]
        axes[0].axhline(market, color="#4B5563", linewidth=1.4, label=f"Market Sharpe ({market:.2f}, {START[:4]}-{END[:4]})")
        axes[1].axhline(1.96, color="#4B5563", linewidth=1.4, linestyle=":", label="t = 1.96")
        for axis, title, ylabel in [(axes[0], "Sharpe Ratio After Costs", "Sharpe ratio"),
                                    (axes[1], "CAPM Alpha t-Statistic After Costs", "Newey-West t of CAPM alpha")]:
            axis.axvline(0.001, color="#9CA3AF", linewidth=1.0, linestyle="--")
            axis.set_title(title, fontsize=12, pad=9)
            axis.set_xlabel("Trading cost per dollar traded")
            axis.set_ylabel(ylabel)
            axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
            axis.legend(frameon=False, fontsize=9)
        fig.suptitle("How Trading Costs Erode the Top-50 Portfolio", fontsize=15, fontweight="bold", y=0.975)
        caption = ("Net return = gross return - cost x (sum of absolute weight changes). Monthly rebalancing; the dashed vertical "
                   f"line marks the 0.10% assumption used elsewhere.\nBoth portfolios: holding months {START} to {END}.")
        fig.text(0.035, 0.03, caption, ha="left", va="bottom", fontsize=8.1, linespacing=1.4)
        fig.tight_layout(rect=(0.025, 0.1, 0.985, 0.93), w_pad=2.6)
        fig.savefig(OUTPUT_DIR / "top50_cost_sensitivity.png", dpi=200, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
