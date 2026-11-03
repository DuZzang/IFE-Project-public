"""CNN plus one, two or all three factors: long-only decile 10 against the S&P 500 (exploratory).

Usage: python src/cnn_plus_factor_subsets.py PATH/TO/sp500_monthly.csv
Exploratory check requested after the professor feedback (2026-09-29): the subsets were compared after
the main results were known, so they are hypotheses, not a pre-specified design. Four-signal matched
universe, ranks recomputed within it, equal weight on CNN and each added factor; buy decile 10 (EW, VW)
or the top 50 (EW); net of 0.10% per dollar traded; holding months 2005-01 to 2019-12.
Only portfolio-level aggregates are written.
"""

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
sys.path.insert(1, str(ROOT / "handoff" / "ivol_combination" / "src"))
from common import assign_deciles  # noqa: E402
from eval_window import clip  # noqa: E402
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402
from longonly_d10_vs_sp500 import metrics  # noqa: E402
from three_factor_combination import build_panel, combined_score, pct_rank  # noqa: E402
from top_n_combination import CNN_SCORES, top_n_returns  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/cnn_plus_factor_subsets"
COST = 0.001
DESIGNS = {  # rank indicators (mom, low beta, low ivol, cnn); weights are equal among the included signals
    "CNN only": (0, 0, 0, 1),
    "CNN + MOM": (1, 0, 0, 1),
    "CNN + LowBeta": (0, 1, 0, 1),
    "CNN + LowIVOL": (0, 0, 1, 1),
    "CNN + MOM + LowBeta": (1, 1, 0, 1),
    "CNN + MOM + LowIVOL": (1, 0, 1, 1),
    "CNN + LowBeta + LowIVOL": (0, 1, 1, 1),
    "CNN + all 3": (1, 1, 1, 1),
}
STYLE = {  # Okabe-Ito hues; each design keeps its colour in both panels
    "CNN only": ("#111111", "-", 2.4),
    "CNN + MOM": ("#0072B2", "-", 1.7),
    "CNN + LowBeta": ("#D55E00", "-", 2.4),
    "CNN + LowIVOL": ("#009E73", "-", 1.7),
    "CNN + MOM + LowBeta": ("#CC79A7", "-", 1.9),
    "CNN + MOM + LowIVOL": ("#56B4E9", "-", 1.7),
    "CNN + LowBeta + LowIVOL": ("#E69F00", "-", 1.9),
    "CNN + all 3": ("#6A4C93", "-", 1.9),
    "S&P 500 TR": ("#7A7A7A", "--", 1.8),
}


def draw(net: pd.DataFrame, summary: pd.DataFrame) -> Path:
    panels = [("CNN + one factor", ["CNN only", "CNN + MOM", "CNN + LowBeta", "CNN + LowIVOL", "S&P 500 TR"]),
              ("CNN + two or three factors", ["CNN only", "CNN + MOM + LowBeta", "CNN + MOM + LowIVOL",
                                              "CNN + LowBeta + LowIVOL", "CNN + all 3", "S&P 500 TR"])]
    dates = pd.PeriodIndex(net.index, freq="M").to_timestamp(how="end")
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.edgecolor": "#6B7280", "axes.labelcolor": "#1F2937",
                         "xtick.color": "#374151", "ytick.color": "#374151", "text.color": "#111827"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True)
        for axis, (title, keys) in zip(axes, panels):
            for key in keys:
                color, style, width = STYLE[key]
                wealth = np.cumprod(1.0 + net[key].to_numpy())
                axis.plot(dates, wealth, color=color, linestyle=style, linewidth=width, label=f"{key}  ${wealth[-1]:.2f}")
            axis.set_yscale("log")
            ticks = [0.5, 1, 2, 3, 5, 8, 12]
            axis.set_yticks(ticks)
            axis.set_yticklabels([f"${t:g}" for t in ticks])
            axis.yaxis.set_minor_formatter(plt.NullFormatter())
            axis.axhline(1.0, color="#9CA3AF", linewidth=0.8, linestyle=":")
            axis.set_title(title, fontsize=12, pad=8)
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
            axis.legend(loc="upper left", frameon=False, fontsize=8.8)
        axes[0].set_ylabel("Growth of $1 (log scale)")
        fig.suptitle("Bonus: CNN + subsets of the three factors, long-only decile 10 (EW, net of 0.10%)",
                     fontsize=13.5, fontweight="bold", y=0.985)
        fig.text(0.02, 0.012, "Holding months 2005-01 to 2019-12; four-signal matched universe; equal weight on CNN and each added "
                 "factor rank. Exploratory: combinations compared after the main results were known.",
                 fontsize=8.2, color="#4B5563")
        fig.tight_layout(rect=(0.01, 0.04, 0.99, 0.95), w_pad=2.0)
        output = OUTPUT_DIR / "cnn_plus_factor_subsets_d10_ew.png"
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)
    return output


def main(sp_path: str) -> None:
    sp = pd.read_csv(sp_path, dtype={"month": str}).set_index("month")["sp500tr_ret"]
    rf = load_french_monthly(FF5_MONTHLY_ZIP)["rf"]
    rf.index = rf.index.astype(str)
    scores = pd.read_csv(CNN_SCORES, usecols=["PERMNO", "formation_month", "cnn_prob_up"], dtype={"formation_month": str})
    panel, *_ = build_panel()
    four = panel.merge(scores, on=["PERMNO", "formation_month"], how="inner", validate="one_to_one")
    g = four.groupby("formation_month")
    ranks = np.column_stack([g["mom"].transform(pct_rank), 1.0 - g["beta"].transform(pct_rank),
                             1.0 - g["ivol_daily"].transform(pct_rank), g["cnn_prob_up"].transform(pct_rank)])

    rows, net = [], {}
    for name, flags in DESIGNS.items():
        weights = np.array(flags, float) / sum(flags)
        scored = four.assign(score=combined_score(list(ranks.T), weights))
        d10 = scored.loc[assign_deciles(scored, signal_column="score").astype(int) == 10]
        for label, frame, n, weighting in [("D10 EW", d10, 10**7, "EW"), ("D10 VW", d10, 10**7, "VW"), ("Top50 EW", scored, 50, "EW")]:
            monthly = clip(top_n_returns(frame, "score", n, weighting)[0])
            monthly.index = monthly.index.astype(str)
            r = monthly["gross"] - COST * monthly["traded"]
            rows.append({"design": name, "portfolio": label, "turnover_one_way": monthly["traded"].mean() / 2,
                         **metrics(r, rf.reindex(r.index), sp.reindex(r.index))})
            if label == "D10 EW":
                net[name] = r
    net = pd.DataFrame(net)
    net["S&P 500 TR"] = sp.reindex(net.index)
    summary = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "cnn_plus_factor_subsets.csv", index=False)
    net.to_csv(OUTPUT_DIR / "d10_ew_net_monthly.csv")
    print(draw(net, summary))
    pd.set_option("display.width", 220)
    show = ["design", "portfolio", "annual_mean", "volatility", "sharpe", "max_drawdown", "alpha_sp", "alpha_sp_t", "wealth"]
    print(summary[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
