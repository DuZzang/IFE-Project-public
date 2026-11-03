"""Two-panel decile figures comparing returns before and after the delisting correction.

Reads the portfolio-level summaries written by ``delisting_robustness.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/delisting_robustness"

COLORS = {"EW": "#0072B2", "VW": "#D55E00"}
MARKERS = {"EW": "o", "VW": "s"}
SPECS = {
    "IVOL": {
        "title": "IVOL Decile Portfolios With Delisting Returns - Professor CSV Universe",
        "xlabel": "IVOL Signal Decile (1 = Low, 10 = High)",
        "period": "Formation 2000-01 to 2020-11; returns 2000-02 to 2020-12 (251 months)",
        "spread": "Q1-Q10",
        "file": "ivol_deciles_with_delisting.png",
    },
    "CNN": {
        "title": "CNN Decile Portfolios With Delisting Returns - Professor CSV Universe",
        "xlabel": "CNN Signal Decile (1 = Lowest P(up), 10 = Highest)",
        "period": "Formation 2001-01 to 2019-12; returns 2001-02 to 2020-01 (228 months)",
        "spread": "Q10-Q1",
        "file": "cnn_deciles_with_delisting.png",
    },
    "MOM": {
        "title": "Momentum Decile Portfolios With Delisting Returns - Professor CSV Universe",
        "xlabel": "Momentum Decile (1 = Losers, 10 = Winners)",
        "period": "Signal P(t-1)/P(t-12) - 1 on exact calendar months. Formation 2001-01 to 2020-11; returns 2001-02 to 2020-12 (239 months)",
        "spread": "Q10-Q1",
        "file": "momentum_deciles_with_delisting.png",
        "label_xy": (4.2, 0.03),
    },
    "LOWBETA": {
        "title": "Low-Beta Decile Portfolios With Delisting Returns - Professor CSV Universe",
        "xlabel": "Beta Decile (1 = Low, 10 = High)",
        "period": "60-month CAPM beta vs Fama-French market (at least 48 months); a decile sort, not the beta-neutral FP BAB. "
        "Formation 2004-12 to 2020-11; returns 2005-01 to 2020-12 (192 months)",
        "spread": "Q1-Q10",
        "file": "lowbeta_deciles_with_delisting.png",
    },
}


def spread_text(spreads: pd.DataFrame, strategy: str) -> str:
    parts = []
    for weighting in ["EW", "VW"]:
        rows = spreads.loc[(spreads["strategy"] == strategy) & (spreads["weighting"] == weighting)]
        before = rows.loc[rows["version"] == "adj_prc_only"].iloc[0]
        after = rows.loc[rows["version"] == "with_delisting"].iloc[0]
        parts.append(
            f"{weighting} {before['spread']}: {before['annualized_mean']:.2%} (NW t {before['newey_west_t_lag4']:.2f})"
            f" -> {after['annualized_mean']:.2%} (NW t {after['newey_west_t_lag4']:.2f})"
        )
    return "; ".join(parts)


def draw(
    strategy: str,
    deciles: pd.DataFrame,
    spreads: pd.DataFrame,
    spec_override: dict | None = None,
    output_dir: Path | None = None,
) -> Path:
    spec = {**SPECS[strategy], **(spec_override or {})}
    data = deciles.loc[deciles["strategy"] == strategy]
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharex=True)
        panels = [
            ("annualized_mean", "Annualized Mean Return", "Realized Return"),
            ("annualized_volatility", "Annualized Volatility", "Realized Volatility"),
        ]
        for axis, (metric, ylabel, panel_title) in zip(axes, panels):
            for weighting in ["EW", "VW"]:
                color = COLORS[weighting]
                for version, style in [
                    ("adj_prc_only", {"linestyle": "--", "markerfacecolor": "white", "alpha": 0.6, "linewidth": 1.6}),
                    ("with_delisting", {"linestyle": "-", "markerfacecolor": color, "alpha": 1.0, "linewidth": 2.2}),
                ]:
                    series = data.loc[
                        (data["weighting"] == weighting) & (data["version"] == version)
                    ].sort_values("decile")
                    axis.plot(
                        series["decile"],
                        series[metric],
                        color=color,
                        marker=MARKERS[weighting],
                        markersize=6,
                        markeredgewidth=1.3,
                        markeredgecolor=color,
                        **style,
                    )
            axis.set_title(panel_title, fontsize=12, pad=9)
            axis.set_xticks(range(1, 11))
            axis.set_xlabel(spec["xlabel"])
            axis.set_ylabel(ylabel)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
            axis.set_axisbelow(True)
        axes[0].axhline(0.0, color="#4B5563", linewidth=0.9, zorder=0)
        axes[1].set_ylim(bottom=0.0)

        # Label the single largest before->after move so the main change reads without the caption.
        returns = data.pivot_table(
            index=["weighting", "decile"], columns="version", values="annualized_mean"
        )
        change = (returns["with_delisting"] - returns["adj_prc_only"]).abs()
        weighting, decile = change.idxmax()
        before = returns.loc[(weighting, decile), "adj_prc_only"]
        after = returns.loc[(weighting, decile), "with_delisting"]
        # Keep the label inside the plot: left of a right-side point, right of a left-side one.
        label_xy = spec.get("label_xy") or (
            (decile - 3.6, after - 0.035) if decile >= 6 else (decile + 1.2, after + 0.002)
        )
        axes[0].annotate(
            f"{weighting} Q{decile}: {before:.1%} -> {after:.1%}",
            xy=(decile, after),
            xytext=label_xy,
            fontsize=9,
            color="#1F2937",
            arrowprops={"arrowstyle": "-", "color": "#6B7280", "linewidth": 0.8},
        )

        handles = [
            Line2D([], [], color=COLORS["EW"], marker="o", linestyle="-", linewidth=2.2, markersize=6),
            Line2D([], [], color=COLORS["EW"], marker="o", linestyle="--", markerfacecolor="white", alpha=0.6, linewidth=1.6, markersize=6),
            Line2D([], [], color=COLORS["VW"], marker="s", linestyle="-", linewidth=2.2, markersize=6),
            Line2D([], [], color=COLORS["VW"], marker="s", linestyle="--", markerfacecolor="white", alpha=0.6, linewidth=1.6, markersize=6),
        ]
        labels = [
            "EW - with delisting returns",
            "EW - ADJ_PRC only (before)",
            "VW - with delisting returns",
            "VW - ADJ_PRC only (before)",
        ]
        fig.suptitle(spec["title"], fontsize=15, fontweight="bold", y=0.975)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=False, fontsize=9.5)
        caption = (
            "Deciles are unchanged; only the realized t+1 return is replaced by ret_final from "
            "handoff/delisting_supplement (CRSP MthRet in delisting months, otherwise professor ADJ_PRC).\n"
            f"{spec['period']}.\n"
            "Mean = 12 x monthly arithmetic mean; volatility = sqrt(12) x monthly sample SD. "
            "Missing returns are excluded, never set to zero.\n"
            f"Spreads: {spread_text(spreads, strategy)}."
        )
        fig.text(0.035, 0.022, caption, ha="left", va="bottom", fontsize=8.1, linespacing=1.4)
        fig.tight_layout(rect=(0.025, 0.2, 0.985, 0.875), w_pad=2.6)
        output = (output_dir or OUTPUT_DIR) / spec["file"]
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)
    return output


def main() -> None:
    decile_files = ["decile_summary.csv", "mom_lowbeta_decile_summary.csv"]
    spread_files = ["spread_summary.csv", "mom_lowbeta_spread_summary.csv"]
    deciles = pd.concat([pd.read_csv(OUTPUT_DIR / f) for f in decile_files if (OUTPUT_DIR / f).exists()])
    spreads = pd.concat([pd.read_csv(OUTPUT_DIR / f) for f in spread_files if (OUTPUT_DIR / f).exists()])
    for strategy in SPECS:
        if strategy in set(deciles["strategy"]):
            print(draw(strategy, deciles, spreads))


if __name__ == "__main__":
    main()
