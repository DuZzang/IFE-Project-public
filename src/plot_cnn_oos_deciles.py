from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description="Plot CNN OOS return and volatility by score decile.")
    result.add_argument(
        "--input-dir",
        type=Path,
        default=root / "outputs" / "cnn_oos_deciles_2001_2019",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "cnn_oos_decile_return_volatility.png",
    )
    return result


def save_figure(data: pd.DataFrame, audit: dict, output: Path) -> None:
    expected = list(range(1, 11))
    if data["decile"].tolist() != expected:
        raise ValueError(f"Expected deciles {expected}; found {data['decile'].tolist()}")
    metric_columns = [column for column in data if column != "decile"]
    if not np.isfinite(data[metric_columns].to_numpy(float)).all():
        raise ValueError("Figure data contains non-finite values")

    palette = {"EW (primary)": "#0072B2", "VW (formation cap)": "#D55E00"}
    styles = {
        "EW (primary)": {"linestyle": "-", "marker": "o", "markerfacecolor": "#0072B2"},
        "VW (formation cap)": {"linestyle": "--", "marker": "s", "markerfacecolor": "white"},
    }
    series = {"EW (primary)": "equal_weight", "VW (formation cap)": "value_weight"}
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
        figure, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharex=True)
        specs = [
            ("annualized_mean_return", "Annualized Mean Return", "Realized Return"),
            ("annualized_volatility", "Annualized Volatility", "Realized Volatility"),
        ]
        handles: list = []
        labels: list[str] = []
        for axis, (metric, ylabel, title) in zip(axes, specs, strict=True):
            for label, prefix in series.items():
                line, = axis.plot(
                    data["decile"],
                    data[f"{prefix}_{metric}"],
                    color=palette[label],
                    linewidth=2.1,
                    markersize=5.8,
                    markeredgewidth=1.3,
                    markeredgecolor=palette[label],
                    label=label,
                    **styles[label],
                )
                if axis is axes[0]:
                    handles.append(line)
                    labels.append(label)
            axis.set_title(title, fontsize=12, pad=9)
            axis.set_xticks(range(1, 11))
            axis.set_xlabel("CNN Score Decile (1 = Lowest P(up), 10 = Highest)")
            axis.set_ylabel(ylabel)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
            axis.set_axisbelow(True)
        axes[0].axhline(0.0, color="#4B5563", linewidth=0.9, zorder=0)
        axes[1].set_ylim(bottom=0.0)

        figure.suptitle(
            "CNN OOS Decile Portfolio Performance - Professor CSV Universe",
            fontsize=15,
            fontweight="bold",
            y=0.975,
        )
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=2,
            frameon=False,
        )
        formation_start, formation_end = audit["formation_period"]
        holding_start, holding_end = audit["holding_period"]
        caption = (
            "Figure. Stocks are sorted each month by the five-model ensemble mean of "
            "P(next-month return > 0); realized returns are not used in sorting.\n"
            "The universe is the professor CSV at formation with EXCHCD 1/2/3. Returns use exact "
            "adjacent-month ADJ_PRC changes; missing outcomes are excluded, never set to zero.\n"
            f"Formation: {formation_start} to {formation_end}; holding returns: {holding_start} to "
            f"{holding_end}. Mean = 12 x monthly mean; volatility = sqrt(12) x monthly sample SD."
        )
        figure.text(0.035, 0.022, caption, ha="left", va="bottom", fontsize=8.1, linespacing=1.4)
        figure.tight_layout(rect=(0.025, 0.175, 0.985, 0.875), w_pad=2.6)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, facecolor="white")
        plt.close(figure)


def main() -> None:
    args = parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    data = pd.read_csv(input_dir / "cnn_decile_figure_data.csv")
    audit = json.loads(
        (input_dir / "cnn_oos_deciles_audit.json").read_text(encoding="utf-8")
    )
    save_figure(data, audit, args.output.expanduser().resolve())
    print(f"Wrote CNN decile figure to {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
