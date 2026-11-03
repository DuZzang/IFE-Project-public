"""Decile figures for the Momentum + BAB + IVOL combined score used by the top-50 portfolio.

Same score as ``top_n_combination`` (monthly rank percentiles: momentum ascending, FP beta and
IVOL descending, 1/3 each), same stocks (all three signals, professor CSV EXCHCD 1/2/3), holding
2005-01 to 2019-12 (``eval_window``), delisting-corrected returns. Decile 1 = worst score, 10 = best.
Figure 1: realized return and volatility by decile (EW, VW).
Figure 2: FF3 and FF5 alphas by decile (official monthly Kenneth French factors).
Only portfolio-level aggregates are written.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_window import PERIOD_TEXT  # noqa: E402
from ivol_specifications import FF3_MONTHLY_ZIP, FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402
from performance_metrics import newey_west_t_stat, ols_alpha_hac  # noqa: E402
from three_factor_combination import add_scores, build_panel, decile_returns  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/three_factor_combination"
COLORS = {"EW": "#0072B2", "VW": "#D55E00"}
MARKERS = {"EW": "o", "VW": "s"}
MODELS = {"FF3": ["mkt_rf", "smb", "hml"], "FF5": ["mkt_rf", "smb", "hml", "rmw", "cma"]}
RC = {"font.family": "DejaVu Serif", "axes.edgecolor": "#6B7280", "axes.labelcolor": "#1F2937",
      "xtick.color": "#374151", "ytick.color": "#374151", "text.color": "#111827"}
BASE_CAPTION = (
    "Score = mean of monthly rank percentiles: momentum P(t-1)/P(t-12)-1 (higher is better), FP beta and IVOL (lower is better). "
    "Same score as the top-50 portfolio.\n"
    f"Stocks with all three signals in the professor CSV universe; {PERIOD_TEXT}; delisting-corrected.\n"
)


def alpha(series: pd.Series, factors: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    x = factors.reindex(series.index)[columns]
    fit = ols_alpha_hac(series.reset_index(drop=True), x.reset_index(drop=True), lag=4)
    return fit["annualized_alpha"], fit["alpha_t_stat_newey_west"]


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = {"FF3": load_french_monthly(FF3_MONTHLY_ZIP), "FF5": load_french_monthly(FF5_MONTHLY_ZIP)}
    panel, *_ = build_panel()
    scored = add_scores(panel)
    stats, alphas = [], []
    for weighting in ["EW", "VW"]:
        wide = decile_returns(scored, "combined_score", weighting)
        spread = (wide[10] - wide[1]).dropna()
        for decile in range(1, 11):
            stats.append({"weighting": weighting, "portfolio": str(decile),
                          "annual_mean": 12 * wide[decile].mean(),
                          "annual_volatility": math.sqrt(12) * wide[decile].std(ddof=1)})
        stats.append({"weighting": weighting, "portfolio": "Q10-Q1", "annual_mean": 12 * spread.mean(),
                      "annual_volatility": math.sqrt(12) * spread.std(ddof=1),
                      "nw_t": newey_west_t_stat(spread, lag=4)})
        for model, columns in MODELS.items():
            rf = factors[model]["rf"].reindex(wide.index)
            for decile in range(1, 11):
                a, t = alpha(wide[decile] - rf, factors[model], columns)
                alphas.append({"weighting": weighting, "model": model, "portfolio": str(decile), "alpha_annual": a, "alpha_t": t})
            a, t = alpha(spread, factors[model], columns)
            alphas.append({"weighting": weighting, "model": model, "portfolio": "Q10-Q1", "alpha_annual": a, "alpha_t": t})
    return pd.DataFrame(stats), pd.DataFrame(alphas)


def style_axis(axis, title: str, ylabel: str | None) -> None:
    axis.set_title(title, fontsize=12, pad=9)
    axis.set_xticks(range(1, 11))
    axis.set_xlabel("Combined Score Decile (1 = Worst, 10 = Best)")
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)


def draw_returns(stats: pd.DataFrame) -> Path:
    deciles = stats.loc[stats["portfolio"] != "Q10-Q1"].assign(decile=lambda d: d["portfolio"].astype(int))
    spread = stats.loc[stats["portfolio"] == "Q10-Q1"].set_index("weighting")
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharex=True)
        for axis, (metric, title, ylabel) in zip(
            axes, [("annual_mean", "Realized Return", "Annualized Mean Return"),
                   ("annual_volatility", "Realized Volatility", "Annualized Volatility")]
        ):
            for weighting in ["EW", "VW"]:
                series = deciles.loc[deciles["weighting"] == weighting].sort_values("decile")
                axis.plot(series["decile"], series[metric], color=COLORS[weighting], marker=MARKERS[weighting],
                          markersize=6, linewidth=2.2, linestyle="-" if weighting == "EW" else "--",
                          markerfacecolor=COLORS[weighting] if weighting == "EW" else "white",
                          markeredgewidth=1.3, label="EW" if weighting == "EW" else "VW (CRSP cap)")
            style_axis(axis, title, ylabel)
        axes[0].axhline(0.0, color="#4B5563", linewidth=0.9, zorder=0)
        axes[1].set_ylim(bottom=0.0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.suptitle("Momentum + BAB + IVOL Combined Score Deciles", fontsize=15, fontweight="bold", y=0.975)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False, fontsize=9.5)
        caption = BASE_CAPTION + (
            "Mean = 12 x monthly arithmetic mean; volatility = sqrt(12) x monthly sample SD. "
            f"Q10-Q1: EW {spread.loc['EW', 'annual_mean']:.2%} (NW t {spread.loc['EW', 'nw_t']:.2f}), "
            f"VW {spread.loc['VW', 'annual_mean']:.2%} (NW t {spread.loc['VW', 'nw_t']:.2f})."
        )
        fig.text(0.035, 0.022, caption, ha="left", va="bottom", fontsize=8.1, linespacing=1.4)
        fig.tight_layout(rect=(0.025, 0.2, 0.985, 0.875), w_pad=2.6)
        output = OUTPUT_DIR / "combination_score_deciles_return_volatility.png"
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)
    return output


def draw_alphas(alphas: pd.DataFrame) -> Path:
    deciles = alphas.loc[alphas["portfolio"] != "Q10-Q1"].assign(decile=lambda d: d["portfolio"].astype(int))
    spreads = alphas.loc[alphas["portfolio"] == "Q10-Q1"].set_index(["model", "weighting"])
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharey=True)
        for axis, (model, title) in zip(axes, [("FF3", "Fama-French 3-Factor Alpha"), ("FF5", "Fama-French 5-Factor Alpha")]):
            for weighting, offset in [("EW", -0.1), ("VW", 0.1)]:
                series = deciles.loc[(deciles["model"] == model) & (deciles["weighting"] == weighting)].sort_values("decile")
                x = series["decile"].to_numpy() + offset
                a, t = series["alpha_annual"].to_numpy(), series["alpha_t"].to_numpy()
                color = COLORS[weighting]
                axis.errorbar(x, a, yerr=1.96 * np.abs(a / t), fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, alpha=0.55)
                axis.plot(x, a, color=color, linewidth=2.0)
                sig = np.abs(t) >= 1.96
                axis.scatter(x[sig], a[sig], marker=MARKERS[weighting], s=46, color=color, zorder=3)
                axis.scatter(x[~sig], a[~sig], marker=MARKERS[weighting], s=46, facecolor="white", edgecolor=color,
                             linewidth=1.4, zorder=3)
                s = spreads.loc[(model, weighting)]
                axis.annotate(f"{weighting} Q10-Q1: {s['alpha_annual']:.1%} (t {s['alpha_t']:.2f})",
                              xy=(0.03, 0.9 if weighting == "EW" else 0.84), xycoords="axes fraction", fontsize=9, color="#1F2937")
            axis.axhline(0.0, color="#4B5563", linewidth=0.9, zorder=0)
            style_axis(axis, title, "Annualized alpha (x12)" if model == "FF3" else None)
        handles = [
            Line2D([], [], color=COLORS["EW"], marker="o", linewidth=2.0, markersize=7),
            Line2D([], [], color=COLORS["VW"], marker="s", linewidth=2.0, markersize=7),
            Line2D([], [], color="#6B7280", marker="o", linestyle="none", markersize=7),
            Line2D([], [], color="#6B7280", marker="o", linestyle="none", markersize=7, markerfacecolor="white"),
        ]
        fig.suptitle("Combined Score Decile Alphas", fontsize=15, fontweight="bold", y=0.975)
        fig.legend(handles, ["EW", "VW (CRSP cap)", "|t| >= 1.96 (filled)", "|t| < 1.96 (hollow)"],
                   loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=False, fontsize=9.5)
        caption = BASE_CAPTION + (
            "Each decile's excess return over RF is regressed on official monthly Kenneth French factors; alpha = 12 x intercept; "
            "whiskers = 95% interval (Newey-West(4))."
        )
        fig.text(0.035, 0.022, caption, ha="left", va="bottom", fontsize=8.1, linespacing=1.4)
        fig.tight_layout(rect=(0.025, 0.2, 0.985, 0.875), w_pad=2.6)
        output = OUTPUT_DIR / "combination_score_decile_alphas.png"
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)
    return output


def main() -> None:
    stats, alphas = compute()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUTPUT_DIR / "combination_score_deciles.csv", index=False)
    alphas.to_csv(OUTPUT_DIR / "combination_score_decile_alphas.csv", index=False)
    pd.set_option("display.width", 220)
    print(stats.pivot_table(index="weighting", columns="portfolio", values=["annual_mean", "annual_volatility"]).mul(100).round(1).to_string())
    cell = alphas.assign(v=lambda d: (100 * d["alpha_annual"]).round(1).astype(str) + " (" + d["alpha_t"].round(2).astype(str) + ")")
    print(cell.pivot_table(index=["weighting", "model"], columns="portfolio", values="v", aggfunc="first").to_string())
    print(draw_returns(stats))
    print(draw_alphas(alphas))


if __name__ == "__main__":
    main()
