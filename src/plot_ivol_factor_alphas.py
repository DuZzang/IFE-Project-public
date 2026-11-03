"""IVOL decile alphas under the Fama-French 3- and 5-factor models.

Each decile's delisting-corrected excess return (ret_final minus RF) is regressed on the
official monthly FF3 and FF5 factors; the intercept is annualized (x12) with a
Newey-West(4) t-statistic. Deciles are the same monthly equal-count IVOL sorts used in
``ivol_specifications.py`` (designs 1-2 all stocks, 5-6 price >= $5).
Only portfolio-level aggregates are written.
"""

from __future__ import annotations

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
from ivol_specifications import (  # noqa: E402
    ASSIGNMENTS,
    FF3_MONTHLY_ZIP,
    FF5_MONTHLY_ZIP,
    SUPPLEMENT,
    assign_groups,
    formation_prices,
    load_french_monthly,
)
from performance_metrics import ols_alpha_hac  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/ivol_specifications"
COLORS = {"EW": "#0072B2", "VW": "#D55E00"}
MARKERS = {"EW": "o", "VW": "s"}
MODELS = {"FF3": ["mkt_rf", "smb", "hml"], "FF5": ["mkt_rf", "smb", "hml", "rmw", "cma"]}
SAMPLES = {
    "all": ("All Stocks", "ivol_decile_alphas_ff3_ff5.png", "Professor CSV universe, all stocks"),
    "price5": ("Price >= $5", "ivol_decile_alphas_ff3_ff5_price5.png", "Formation-month CRSP |MthPrc| >= $5"),
}


def decile_returns(sample: pd.DataFrame, weighting: str) -> pd.DataFrame:
    usable = sample.dropna(subset=["ret_final"])
    if weighting == "VW":
        usable = usable.loc[usable["formation_market_cap"].gt(0)].assign(
            weighted=lambda d: d["ret_final"] * d["formation_market_cap"]
        )
        sums = usable.groupby(["holding_month", "group"])[["weighted", "formation_market_cap"]].sum()
        return (sums["weighted"] / sums["formation_market_cap"]).unstack("group").sort_index()
    return usable.groupby(["holding_month", "group"])["ret_final"].mean().unstack("group").sort_index()


def fit(series: pd.Series, factors: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    x = factors.reindex(series.index)[columns]
    result = ols_alpha_hac(series.reset_index(drop=True), x.reset_index(drop=True), lag=4)
    return result["annualized_alpha"], result["alpha_t_stat_newey_west"]


def compute(frame: pd.DataFrame, factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sample_key in SAMPLES:
        sample = frame.loc[frame["price"].ge(5.0)] if sample_key == "price5" else frame
        sample = sample.assign(group=assign_groups(sample, 10, "all"))
        for weighting in ["EW", "VW"]:
            wide = decile_returns(sample, weighting)
            for model, columns in MODELS.items():
                rf = factors[model]["rf"].reindex(wide.index)
                for decile in range(1, 11):
                    alpha, t = fit(wide[decile] - rf, factors[model], columns)
                    rows.append((sample_key, weighting, model, str(decile), alpha, t))
                alpha, t = fit(wide[1] - wide[10], factors[model], columns)
                rows.append((sample_key, weighting, model, "Q1-Q10", alpha, t))
    return pd.DataFrame(rows, columns=["sample", "weighting", "model", "portfolio", "alpha_annual", "alpha_t"])


def draw(
    alphas: pd.DataFrame,
    sample_key: str,
    period: str = "formed 2000-01 to 2020-11, returns 2000-02 to 2020-12 (251 months)",
    output_dir: Path | None = None,
) -> Path:
    title_suffix, filename, universe = SAMPLES[sample_key]
    data = alphas.loc[alphas["sample"] == sample_key]
    deciles = data.loc[data["portfolio"] != "Q1-Q10"].assign(decile=lambda d: d["portfolio"].astype(int))
    spreads = data.loc[data["portfolio"] == "Q1-Q10"].set_index(["model", "weighting"])

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
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharey=True)
        for axis, (model, panel_title) in zip(
            axes, [("FF3", "Fama-French 3-Factor Alpha"), ("FF5", "Fama-French 5-Factor Alpha")]
        ):
            for weighting, offset in [("EW", -0.1), ("VW", 0.1)]:
                series = deciles.loc[(deciles["model"] == model) & (deciles["weighting"] == weighting)].sort_values("decile")
                x = series["decile"].to_numpy() + offset
                alpha = series["alpha_annual"].to_numpy()
                t = series["alpha_t"].to_numpy()
                half_width = 1.96 * np.abs(alpha / t)
                color = COLORS[weighting]
                axis.errorbar(x, alpha, yerr=half_width, fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, alpha=0.55)
                axis.plot(x, alpha, color=color, linewidth=2.0)
                significant = np.abs(t) >= 1.96
                axis.scatter(x[significant], alpha[significant], marker=MARKERS[weighting], s=46, color=color, zorder=3)
                axis.scatter(
                    x[~significant], alpha[~significant], marker=MARKERS[weighting], s=46,
                    facecolor="white", edgecolor=color, linewidth=1.4, zorder=3,
                )
                spread = spreads.loc[(model, weighting)]
                axis.annotate(
                    f"{weighting} Q1-Q10: {spread['alpha_annual']:.1%} (t {spread['alpha_t']:.2f})",
                    xy=(0.03, 0.06 if weighting == "EW" else 0.12),
                    xycoords="axes fraction",
                    fontsize=9,
                    color="#1F2937",
                )
            axis.axhline(0.0, color="#4B5563", linewidth=0.9, zorder=0)
            axis.set_title(panel_title, fontsize=12, pad=9)
            axis.set_xticks(range(1, 11))
            axis.set_xlabel("IVOL Signal Decile (1 = Low, 10 = High)")
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
            axis.set_axisbelow(True)
        axes[0].set_ylabel("Annualized alpha (x12)")

        handles = [
            Line2D([], [], color=COLORS["EW"], marker="o", linewidth=2.0, markersize=7),
            Line2D([], [], color=COLORS["VW"], marker="s", linewidth=2.0, markersize=7),
            Line2D([], [], color="#6B7280", marker="o", linestyle="none", markersize=7),
            Line2D([], [], color="#6B7280", marker="o", linestyle="none", markersize=7, markerfacecolor="white"),
        ]
        labels = ["EW", "VW (CRSP cap)", "|t| >= 1.96 (filled)", "|t| < 1.96 (hollow)"]
        fig.suptitle(f"IVOL Decile Alphas - {title_suffix}", fontsize=15, fontweight="bold", y=0.975)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=False, fontsize=9.5)
        caption = (
            f"{universe}; monthly equal-count IVOL deciles {period}.\n"
            "Returns are delisting-corrected (ret_final). Each decile's excess return over RF is regressed on official monthly "
            "Kenneth French factors (202607 file).\n"
            "Alpha = 12 x monthly intercept; whiskers = 95% interval from Newey-West(4) standard errors; "
            "EW and VW points are offset slightly for readability.\n"
            "Q1-Q10 labels give the low-minus-high alpha of the zero-investment spread."
        )
        fig.text(0.035, 0.022, caption, ha="left", va="bottom", fontsize=8.1, linespacing=1.4)
        fig.tight_layout(rect=(0.025, 0.2, 0.985, 0.875), w_pad=2.6)
        output = (output_dir or OUTPUT_DIR) / filename
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)
    return output


def main() -> None:
    frame = pd.read_csv(
        ASSIGNMENTS,
        usecols=["PERMNO", "formation_month", "EXCHCD", "ivol_daily", "formation_market_cap"],
        dtype={"formation_month": str},
    )
    supplement = pd.read_csv(
        SUPPLEMENT, usecols=["PERMNO", "formation_month", "holding_month", "ret_final"],
        dtype={"formation_month": str, "holding_month": str},
    )
    frame = frame.merge(supplement, on=["PERMNO", "formation_month"], how="left", validate="one_to_one")
    frame = formation_prices(frame)
    factors = {"FF3": load_french_monthly(FF3_MONTHLY_ZIP), "FF5": load_french_monthly(FF5_MONTHLY_ZIP)}

    alphas = compute(frame, factors)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    alphas.to_csv(OUTPUT_DIR / "ivol_decile_alphas.csv", index=False)
    table = alphas.assign(cell=lambda d: (100 * d["alpha_annual"]).round(1).astype(str) + " (" + d["alpha_t"].round(2).astype(str) + ")")
    print(table.pivot_table(index=["sample", "weighting", "model"], columns="portfolio", values="cell", aggfunc="first").to_string())
    for sample_key in SAMPLES:
        print(draw(alphas, sample_key))


if __name__ == "__main__":
    main()
