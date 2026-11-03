"""Figure for the Frazzini-Pedersen BAB factor: growth of $1 and ex-ante leg betas."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_window import END, PERIOD_TEXT, START  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/fp_bab"


def main() -> None:
    monthly = pd.read_csv(OUTPUT_DIR / "fp_bab_monthly.csv", dtype={"holding_month": str})
    monthly = monthly.loc[monthly["holding_month"].between(START, END)].reset_index(drop=True)
    summary = pd.read_csv(OUTPUT_DIR / "fp_bab_summary.csv")
    dates = pd.PeriodIndex(monthly["holding_month"], freq="M").to_timestamp(how="end")
    full = summary.loc[summary["period"] == f"{START}~{END}"].set_index("version")

    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "axes.edgecolor": "#6B7280",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.8))

        wealth = axes[0]
        series = [
            ("bab_with_delisting", "FP BAB - with delisting returns", {"color": "#0072B2", "linewidth": 2.2}),
            ("bab_adj_prc_only", "FP BAB - ADJ_PRC only (before)", {"color": "#0072B2", "linewidth": 1.6, "linestyle": "--", "alpha": 0.6}),
            ("mkt_excess", "Market excess return (Mkt-RF), reference", {"color": "#6B7280", "linewidth": 1.4}),
        ]
        for column, label, style in series:
            growth = np.cumprod(1.0 + monthly[column].to_numpy())
            wealth.plot(dates, np.concatenate(([1.0], growth))[1:], label=label, **style)
        wealth.set_yscale("log")
        ticks = [0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]
        wealth.set_yticks(ticks)
        wealth.set_yticklabels([f"${t:g}" for t in ticks])
        wealth.yaxis.set_minor_formatter(plt.NullFormatter())
        wealth.axhline(1.0, color="#4B5563", linewidth=0.8, linestyle=":")
        wealth.set_title("Growth of $1", fontsize=12, pad=9)
        wealth.set_ylabel("Cumulative value (log scale)")
        wealth.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
        wealth.legend(loc="upper left", frameon=False, fontsize=9)
        final = np.prod(1.0 + monthly["bab_with_delisting"])
        wealth.annotate(
            f"${final:.2f}",
            xy=(dates[-1], final),
            xytext=(-6, 8),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color="#1F2937",
        )

        legs = axes[1]
        legs.plot(dates, monthly["beta_high"], color="#D55E00", linewidth=2.0, label="High-beta leg (short), ex-ante beta")
        legs.plot(dates, monthly["beta_low"], color="#009E73", linewidth=2.0, label="Low-beta leg (long), ex-ante beta")
        legs.axhline(1.0, color="#4B5563", linewidth=0.8, linestyle=":")
        legs.set_ylim(0.0, max(2.0, float(monthly["beta_high"].max()) + 0.2))
        legs.set_title("Leg Betas Before Scaling to 1", fontsize=12, pad=9)
        legs.set_ylabel("Shrunk beta (0.6 x estimate + 0.4)")
        legs.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
        legs.legend(loc="upper left", frameon=False, fontsize=9)
        for axis in axes:
            axis.set_xlabel("Holding month")

        fig.suptitle(
            "Frazzini-Pedersen Betting-Against-Beta - Professor CSV Universe",
            fontsize=15,
            fontweight="bold",
            y=0.975,
        )
        after, before = full.loc["with_delisting"], full.loc["adj_prc_only"]
        caption = (
            "Beta = 0.6 x (corr of 3-day log returns over 60 months x stock vol / market vol over 12 months of daily log returns) + 0.4; "
            "market = Fama-French Mkt.\n"
            "Rank-weighted low- and high-beta legs, each scaled to beta 1: r_BAB = (r_L - rf) / beta_L - (r_H - rf) / beta_H. "
            f"{PERIOD_TEXT[0].upper() + PERIOD_TEXT[1:]}.\n"
            f"With delisting returns: {after['annualized_mean']:.2%} a year, volatility {after['annualized_volatility']:.2%}, "
            f"Sharpe {after['sharpe']:.2f}, NW t {after['newey_west_t_lag4']:.2f}, CAPM alpha {after['capm_alpha_annualized']:.2%} "
            f"(t {after['capm_alpha_t_nw4']:.2f}), realized beta {after['realized_market_beta']:.2f}.\n"
            f"ADJ_PRC only: {before['annualized_mean']:.2%} a year, Sharpe {before['sharpe']:.2f}, NW t {before['newey_west_t_lag4']:.2f}. "
            "No transaction or funding costs above the risk-free rate."
        )
        fig.text(0.035, 0.022, caption, ha="left", va="bottom", fontsize=8.1, linespacing=1.4)
        fig.tight_layout(rect=(0.025, 0.2, 0.985, 0.94), w_pad=2.6)
        output = OUTPUT_DIR / "fp_bab.png"
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
