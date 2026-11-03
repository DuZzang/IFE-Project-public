"""COVID 2020 stress test, reported separately from the 2005-01 to 2019-12 evaluation window.

2020 is outside the evaluation window (``eval_window``) and is shown only as a stress test: the
results were already known when the window was chosen, so this is not out-of-sample validation.
Signals and portfolio rules are unchanged; the combination panel is simply extended to formation
2020-11. CNN is absent because the author images stop at the 2019-12 formation month.

Reported per strategy (delisting-corrected returns, compounded):
  January 2020, February-March (crash), April-December (rebound), full 2020, full 2009 (the
  in-sample year of the same crash-then-rebound type), the median calendar year 2005-2019, and the
  rank of 2020 among the 16 calendar years 2005-2020 (1 = worst).
Only portfolio-level aggregates are written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import assign_deciles  # noqa: E402
from delisting_robustness import IVOL, SUPPLEMENT, monthly_portfolios, tradable_at_formation  # noqa: E402
from eval_window import FIRST_FORMATION  # noqa: E402
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402
from momentum_lowbeta_deciles import CRSP_MONTHLY, attach_outcomes, load_prices, momentum_signals  # noqa: E402
from professor_universe import load_formation_caps  # noqa: E402
from three_factor_combination import add_scores, build_panel  # noqa: E402
from top_n_combination import top_n_returns  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/stress_test_2020"
LAST_FORMATION_2020 = "2020-11"
LABELS = {
    "top50_net": "Top 50, 3 signals (net of 0.1%)",
    "top50_gross": "Top 50, 3 signals (before costs)",
    "d10_ew_net": "Decile 10, 3 signals, EW (net of 0.1%)",
    "d10_vw_net": "Decile 10, 3 signals, VW (net of 0.1%)",
    "market": "Market (Fama-French)",
    "universe_ew": "All stocks with signals (EW)",
    "ivol_ew": "IVOL low - high (EW)",
    "ivol_vw": "IVOL low - high (VW)",
    "mom_ew": "Momentum winners - losers (EW)",
    "mom_vw": "Momentum winners - losers (VW)",
    "combined_ew": "Combined score D10 - D1 (EW)",
    "fp_bab": "FP BAB",
}


def compound(series: pd.Series) -> float:
    return float((1.0 + series.dropna()).prod() - 1.0)


def monthly_series() -> pd.DataFrame:
    factors = load_french_monthly(FF5_MONTHLY_ZIP)
    supplement = pd.read_csv(SUPPLEMENT, usecols=["PERMNO", "formation_month", "ret_final"], dtype={"formation_month": str})
    out = {}

    ivol = pd.read_csv(
        IVOL, usecols=["PERMNO", "formation_month", "ivol_daily", "next_month", "formation_market_cap"],
        dtype={"formation_month": str, "next_month": str},
    ).rename(columns={"next_month": "holding_month", "formation_market_cap": "cap"})
    ivol = tradable_at_formation(ivol)  # same eligibility rule as the combinations; deciles re-assigned after it
    ivol["decile"] = assign_deciles(ivol, signal_column="ivol_daily").astype(int)
    ivol = ivol.merge(supplement, on=["PERMNO", "formation_month"], how="left")
    for weighting, weighted in [("ew", False), ("vw", True)]:
        wide = monthly_portfolios(ivol, "ret_final", weighted)
        out[f"ivol_{weighting}"] = wide[1] - wide[10]

    prices, universe = load_prices()
    momentum = tradable_at_formation(momentum_signals(prices, universe)["MOM"])
    caps, _ = load_formation_caps(CRSP_MONTHLY, momentum[["PERMNO", "formation_month"]], chunksize=500_000)
    momentum = attach_outcomes(momentum, supplement, caps)
    for weighting, weighted in [("ew", False), ("vw", True)]:
        wide = monthly_portfolios(momentum, "ret_final", weighted)
        out[f"mom_{weighting}"] = wide[10] - wide[1]

    # FP BAB comes from fp_bab.py, which still has the pre-review missing-return reweighting (open item)
    bab = pd.read_csv(ROOT / "outputs/fp_bab/fp_bab_monthly.csv", dtype={"holding_month": str}).set_index("holding_month")
    out["fp_bab"] = bab["bab_with_delisting"]

    panel, *_ = build_panel(FIRST_FORMATION, LAST_FORMATION_2020)
    scored = add_scores(panel)
    deciles = scored.assign(decile=assign_deciles(scored, signal_column="combined_score").astype(int))
    wide = monthly_portfolios(deciles, "ret_final", False)
    out["combined_ew"] = wide[10] - wide[1]
    top, _ = top_n_returns(scored, "combined_score", 50, "EW")
    out["top50_gross"], out["top50_net"] = top["gross"], top["net"]
    d10 = deciles.loc[deciles["decile"].eq(10)]
    for weighting in ["EW", "VW"]:
        out[f"d10_{weighting.lower()}_net"] = top_n_returns(d10, "combined_score", 10**7, weighting)[0]["net"]
    usable = scored.dropna(subset=["ret_final"])
    out["universe_ew"] = usable.groupby("holding_month")["ret_final"].mean()
    out["market"] = factors["mkt_rf"] + factors["rf"]

    frame = pd.DataFrame({k: v.rename(index=str) for k, v in out.items()}).sort_index()
    return frame.loc[(frame.index >= "2005-01") & (frame.index <= "2020-12")]


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    years = pd.DataFrame({year: frame.loc[f"{year}-01":f"{year}-12"].apply(compound) for year in range(2005, 2021)}).T
    return pd.DataFrame({
        "jan_2020": frame.loc["2020-01"],
        "feb_mar_2020": frame.loc["2020-02":"2020-03"].apply(compound),
        "apr_dec_2020": frame.loc["2020-04":"2020-12"].apply(compound),
        "year_2020": years.loc[2020],
        "year_2009": years.loc[2009],
        "median_year_2005_2019": years.loc[2005:2019].median(),
        "rank_2020_of_16_worst_first": years.rank().loc[2020].astype(int),
        "worst_year_2005_2019": years.loc[2005:2019].idxmin(),
    })


def draw(frame: pd.DataFrame) -> Path:
    year = frame.loc["2019-12":"2020-12"].copy()
    year.iloc[0] = 0.0  # start every line at 1 at the end of 2019
    dates = pd.PeriodIndex(year.index, freq="M").to_timestamp(how="end")
    panels = [
        ("Long-only portfolios: growth of $1 in 2020", [
            ("top50_net", "#D55E00", "-", 2.6), ("market", "#4B5563", "-", 1.8), ("universe_ew", "#0072B2", "--", 1.6)]),
        ("Long-short spreads: growth of $1 in 2020", [
            ("combined_ew", "#D55E00", "-", 2.6), ("ivol_ew", "#CC79A7", "--", 1.8), ("mom_ew", "#0072B2", "--", 1.8),
            ("fp_bab", "#009E73", "-", 1.8)]),
    ]
    with plt.rc_context({"font.family": "DejaVu Serif", "axes.edgecolor": "#6B7280", "axes.labelcolor": "#1F2937",
                         "xtick.color": "#374151", "ytick.color": "#374151", "text.color": "#111827"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 7.4))
        for axis, (title, lines) in zip(axes, panels):
            crash = pd.Timestamp("2020-01-31"), pd.Timestamp("2020-03-31")
            axis.axvspan(*crash, color="#E5E7EB", zorder=0)
            axis.axvline(pd.Timestamp("2020-10-31"), color="#9CA3AF", linewidth=1.0, linestyle=":")
            for key, color, style, width in lines:
                wealth = (1.0 + year[key]).cumprod()
                axis.plot(dates, wealth, color=color, linestyle=style, linewidth=width, marker="o", markersize=3.5,
                          label=f"{LABELS[key]}: {compound(frame.loc['2020-01':'2020-12', key]):+.0%}")
            axis.axhline(1.0, color="#4B5563", linewidth=0.8, linestyle=":")
            axis.set_title(title, fontsize=12, pad=9)
            axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=1, frameon=False, fontsize=8.8)
            axis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
            ymin, ymax = axis.get_ylim()
            axis.text(pd.Timestamp("2020-02-29"), ymax - 0.02 * (ymax - ymin), "Feb-Mar\ncrash", ha="center", va="top", fontsize=8.4,
                      color="#374151")
            axis.text(pd.Timestamp("2020-10-26"), ymax - 0.02 * (ymax - ymin), "Nov: vaccine\nnews", ha="right", va="top",
                      fontsize=8.4, color="#374151")
        fig.suptitle("COVID 2020 Stress Test (outside the 2005-2019 evaluation window)", fontsize=15, fontweight="bold", y=0.975)
        caption = (
            "Same signals and rules as the 2005-2019 evaluation; the combination is simply extended to formation 2020-11. "
            "Delisting-corrected returns, compounded from the end of 2019.\n"
            "Top 50: equal-weighted 50 best combined scores (momentum, FP beta, IVOL; 1/3 each), net of 0.10% per dollar traded.\n"
            "Spreads are zero-investment. CNN is absent (images end at the 2019-12 formation month)."
        )
        fig.text(0.035, 0.02, caption, ha="left", va="bottom", fontsize=8.1, linespacing=1.4)
        fig.tight_layout(rect=(0.025, 0.1, 0.985, 0.93), w_pad=2.6)
        output = OUTPUT_DIR / "stress_2020.png"
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--draw-only" in sys.argv:  # redraw from the saved monthly series
        print(draw(pd.read_csv(OUTPUT_DIR / "monthly_2005_2020.csv", index_col=0).rename(index=str)))
        return
    frame = monthly_series()
    summary = summarize(frame)
    frame.to_csv(OUTPUT_DIR / "monthly_2005_2020.csv")
    summary.to_csv(OUTPUT_DIR / "stress_2020_summary.csv")
    pd.set_option("display.width", 220)
    pct = ["jan_2020", "feb_mar_2020", "apr_dec_2020", "year_2020", "year_2009", "median_year_2005_2019"]
    print((summary[pct] * 100).round(1).join(summary[["rank_2020_of_16_worst_first", "worst_year_2005_2019"]]).to_string())
    print((frame.loc["2020-01":"2020-12"] * 100).round(1).to_string())
    print(draw(frame))


if __name__ == "__main__":
    main()
