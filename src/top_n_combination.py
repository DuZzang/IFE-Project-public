"""Long-only top-N portfolios on the Momentum + BAB + IVOL combined score.

Pre-specified on 2026-09-24 before running; every design below is reported.

Score: the three-signal combined score from ``three_factor_combination`` (monthly rank
percentiles of momentum P(t-1)/P(t-12)-1 ascending, FP beta and IVOL descending, averaged).
Portfolio: at the end of each formation month t, buy the top N stocks by score (ties by
PERMNO), hold for month t+1, rebalance monthly. A holding without a t+1 return (after the
delisting correction) is dropped and the remaining weights are rescaled.

Designs (one change at a time from the primary):
  P   top 50, equal weight, all stocks            (primary)
  S1  top 100, equal weight, all stocks
  S2  top 50, value weight (CRSP MthCap), all stocks
  S3  top 50, equal weight, formation-month price >= $5
  C4  top 50, equal weight, all stocks, four signals (adds CNN P(up) rank, 1/4 each);
      (CNN scores end at formation 2019-12, which the common window already respects)
Benchmarks: Fama-French market (Mkt-RF + RF), and the equal- and value-weighted average of
all stocks with the three signals.
Metrics: arithmetic and compound annual return, volatility, Sharpe (excess of RF), maximum
drawdown from a starting value of 1, CAPM beta and alpha, FF5 alpha (Newey-West(4) t),
one-way turnover, and returns net of 0.10% per dollar traded.
Holding months 2005-01 to 2019-12 (180, ``eval_window``). Only portfolio-level aggregates are written.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402
from performance_metrics import maximum_drawdown, newey_west_t_stat, ols_alpha_hac  # noqa: E402
from three_factor_combination import add_scores, build_panel, combined_score, pct_rank  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/top_n_combination"
CNN_SCORES = ROOT / "outputs/cnn_oos_deciles_2001_2019/cnn_signal_scores_2001_2019.csv.gz"
COST_PER_DOLLAR = 0.001
FF5 = ["mkt_rf", "smb", "hml", "rmw", "cma"]


def top_n_returns(frame: pd.DataFrame, score: str, n: int, weighting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = frame.sort_values(["formation_month", score, "PERMNO"], ascending=[True, False, True])
    top = ranked.groupby("formation_month", sort=True).head(n).copy()
    if weighting == "VW":
        top = top.loc[top["cap"].gt(0)].copy()
        top["weight"] = top["cap"] / top.groupby("formation_month")["cap"].transform("sum")
    else:
        top["weight"] = 1.0 / top.groupby("formation_month")["PERMNO"].transform("size")

    records, previous = [], None
    for month, holdings in top.groupby("formation_month", sort=True):
        target = holdings.set_index("PERMNO")["weight"]
        returns = holdings.set_index("PERMNO")["ret_final"]
        available = returns.notna()
        gross = float((target[available] * returns[available]).sum() / target[available].sum())
        if previous is None:
            traded = 1.0
        else:
            union = target.index.union(previous.index)
            traded = float((target.reindex(union, fill_value=0.0) - previous.reindex(union, fill_value=0.0)).abs().sum())
        drifted = (target[available] * (1.0 + returns[available]))
        previous = drifted / drifted.sum()
        records.append(
            {
                "holding_month": holdings["holding_month"].iloc[0],
                "gross": gross,
                "traded": traded,
                "net": gross - COST_PER_DOLLAR * traded,
                "holdings": int(len(holdings)),
                "missing_returns": int((~available).sum()),
            }
        )
    return pd.DataFrame(records).set_index("holding_month"), top


def summarize(series: pd.Series, factors: pd.DataFrame) -> dict:
    rf = factors["rf"].reindex(series.index)
    excess = series - rf
    capm = ols_alpha_hac(excess.reset_index(drop=True), factors.reindex(series.index)[["mkt_rf"]].reset_index(drop=True), lag=4)
    ff5 = ols_alpha_hac(excess.reset_index(drop=True), factors.reindex(series.index)[FF5].reset_index(drop=True), lag=4)
    beta = float(np.polyfit(factors.reindex(series.index)["mkt_rf"], excess, 1)[0])
    wealth = float(np.prod(1.0 + series))
    return {
        "months": int(len(series)),
        "annual_return": 12 * series.mean(),
        "cagr": wealth ** (12 / len(series)) - 1,
        "volatility": math.sqrt(12) * series.std(ddof=1),
        "sharpe": math.sqrt(12) * excess.mean() / excess.std(ddof=1),
        "max_drawdown": maximum_drawdown(series),
        "final_value_of_1": wealth,
        "capm_beta": beta,
        "capm_alpha": capm["annualized_alpha"],
        "capm_alpha_t": capm["alpha_t_stat_newey_west"],
        "ff5_alpha": ff5["annualized_alpha"],
        "ff5_alpha_t": ff5["alpha_t_stat_newey_west"],
        "excess_mean_t": newey_west_t_stat(excess, lag=4),
    }


def characteristics(top: pd.DataFrame, universe: pd.DataFrame) -> dict:
    ranks = universe.assign(size_pct=universe.groupby("formation_month")["cap"].transform(pct_rank))
    held = top.merge(ranks[["PERMNO", "formation_month", "size_pct"]], on=["PERMNO", "formation_month"], how="left")
    return {
        "avg_size_percentile": float(held["size_pct"].mean()),
        "avg_fp_beta": float(held["beta"].mean()),
        "median_price": float(held["price"].median()),
        "avg_ivol_percentile": float(1.0 - held["ivol_score"].mean()),
        "avg_momentum_percentile": float(held["mom_score"].mean()),
    }


def main() -> None:
    factors = load_french_monthly(FF5_MONTHLY_ZIP)
    panel, *_ = build_panel()
    scored = add_scores(panel)
    scored_price5 = add_scores(panel.loc[panel["price"].ge(5.0)])

    cnn = pd.read_csv(CNN_SCORES, usecols=["PERMNO", "formation_month", "cnn_prob_up"], dtype={"formation_month": str})
    four = scored.merge(cnn, on=["PERMNO", "formation_month"], how="inner", validate="one_to_one")
    four["cnn_score"] = four.groupby("formation_month")["cnn_prob_up"].transform(pct_rank)
    for column in ["mom_score", "beta_score", "ivol_score"]:  # re-rank within the four-signal sample
        source = {"mom_score": "mom", "beta_score": "beta", "ivol_score": "ivol_daily"}[column]
        ranks = four.groupby("formation_month")[source].transform(pct_rank)
        four[column] = ranks if column == "mom_score" else 1.0 - ranks
    four["four_score"] = combined_score([four[c] for c in ["mom_score", "beta_score", "ivol_score", "cnn_score"]], [0.25] * 4)

    designs = {
        "P  top50 EW all": (scored, "combined_score", 50, "EW"),
        "S1 top100 EW all": (scored, "combined_score", 100, "EW"),
        "S2 top50 VW all": (scored, "combined_score", 50, "VW"),
        "S3 top50 EW price>=$5": (scored_price5, "combined_score", 50, "EW"),
        "C4 top50 EW 4 signals": (four, "four_score", 50, "EW"),
    }
    rows, series_store, chars = [], {}, {}
    for label, (frame, score, n, weighting) in designs.items():
        monthly, top = top_n_returns(frame, score, n, weighting)
        series_store[label] = monthly
        chars[label] = characteristics(top, frame)
        for kind in ["gross", "net"]:
            rows.append({"portfolio": label, "returns": kind, **summarize(monthly[kind], factors),
                         "avg_one_way_turnover": monthly["traded"].iloc[1:].mean() / 2,
                         "missing_returns": int(monthly["missing_returns"].sum())})

    months = series_store["P  top50 EW all"].index
    universe = scored.dropna(subset=["ret_final"])
    universe_ew = universe.groupby("holding_month")["ret_final"].mean().reindex(months)
    vw = universe.loc[universe["cap"].gt(0)]
    universe_vw = (vw["ret_final"] * vw["cap"]).groupby(vw["holding_month"]).sum() / vw.groupby("holding_month")["cap"].sum()
    market = (factors["mkt_rf"] + factors["rf"]).reindex(months)
    benchmarks = {"Market (FF Mkt)": market, "Universe EW": universe_ew, "Universe VW": universe_vw.reindex(months)}
    for label, series in benchmarks.items():
        rows.append({"portfolio": label, "returns": "benchmark", **summarize(series, factors)})

    result = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "top_n_results.csv", index=False)
    pd.DataFrame(chars).T.to_csv(OUTPUT_DIR / "top_n_holdings_characteristics.csv")
    pd.concat({k: v for k, v in series_store.items()}, axis=1).to_csv(OUTPUT_DIR / "top_n_monthly.csv")
    pd.DataFrame(benchmarks).to_csv(OUTPUT_DIR / "benchmarks_monthly.csv")

    pd.set_option("display.width", 250)
    show = ["portfolio", "returns", "months", "annual_return", "cagr", "volatility", "sharpe", "max_drawdown",
            "final_value_of_1", "capm_beta", "capm_alpha", "capm_alpha_t", "ff5_alpha", "ff5_alpha_t", "avg_one_way_turnover"]
    print(result[show].round(3).to_string(index=False))
    print(pd.DataFrame(chars).T.round(3).to_string())
    draw(series_store, benchmarks)


def draw(series_store: dict, benchmarks: dict) -> None:
    lines = [
        ("P  top50 EW all", "Top 50, 3 signals (EW, net of costs)", "#D55E00", "-", 2.6, "net"),
        ("C4 top50 EW 4 signals", "Top 50, 4 signals with CNN (EW, net)", "#CC79A7", "--", 1.6, "net"),
        ("Universe EW", "All stocks with signals (EW)", "#0072B2", "--", 1.6, None),
        ("Market (FF Mkt)", "Market (Fama-French)", "#4B5563", "-", 1.6, None),
    ]
    with plt.rc_context({"font.family": "DejaVu Serif", "axes.edgecolor": "#6B7280", "axes.labelcolor": "#1F2937",
                         "xtick.color": "#374151", "ytick.color": "#374151", "text.color": "#111827"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), gridspec_kw={"width_ratios": [1.35, 1]})
        wealth_axis, drawdown_axis = axes
        for key, label, color, style, width, kind in lines:
            series = series_store[key][kind] if kind else benchmarks[key].dropna()
            dates = pd.PeriodIndex(series.index, freq="M").to_timestamp(how="end")
            wealth = np.cumprod(1.0 + series.to_numpy())
            peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
            wealth_axis.plot(dates, wealth, color=color, linestyle=style, linewidth=width, label=label)
            drawdown_axis.plot(dates, wealth / peak - 1.0, color=color, linestyle=style, linewidth=width)
        wealth_axis.set_yscale("log")
        ticks = [0.5, 1, 2, 3, 4, 6]
        wealth_axis.set_yticks(ticks)
        wealth_axis.set_yticklabels([f"${t:g}" for t in ticks])
        wealth_axis.yaxis.set_minor_formatter(plt.NullFormatter())
        wealth_axis.axhline(1.0, color="#4B5563", linewidth=0.8, linestyle=":")
        wealth_axis.set_title("Growth of $1", fontsize=12, pad=9)
        drawdown_axis.set_title("Drawdown", fontsize=12, pad=9)
        drawdown_axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        for axis in axes:
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
            axis.set_xlabel("Holding month")
        handles, labels = wealth_axis.get_legend_handles_labels()
        fig.suptitle("Top-50 Combination Portfolio vs Market", fontsize=15, fontweight="bold", y=0.975)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False, fontsize=9.5)
        caption = (
            "Each month, buy the 50 highest-scoring stocks (equal weight) and hold one month. 3 signals: momentum P(t-1)/P(t-12)-1, "
            "FP beta (low), IVOL (low), 1/3 each.\n"
            "4 signals adds CNN P(up), 1/4 each. Delisting-corrected returns; net line deducts 0.10% "
            "per dollar traded. Holding 2005-01 to 2019-12 (180 months).\n"
            "Benchmarks: Fama-French market (Mkt-RF + RF) and the equal-weighted average of all stocks with the three signals."
        )
        fig.text(0.035, 0.022, caption, ha="left", va="bottom", fontsize=8.1, linespacing=1.4)
        fig.tight_layout(rect=(0.025, 0.17, 0.985, 0.86), w_pad=2.6)
        fig.savefig(OUTPUT_DIR / "top50_vs_market.png", dpi=200, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
