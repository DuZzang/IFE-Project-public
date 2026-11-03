from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from common import ensure_parent, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate IVOL decile portfolios.")
    result.add_argument("--input", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--log", required=True)
    return result


def portfolio_summary(monthly: pd.DataFrame, count_means: dict[str, float]) -> pd.DataFrame:
    records = []
    for portfolio, values in monthly.groupby("portfolio")["return"]:
        values = values.dropna().sort_index()
        n_months = len(values)
        mean_monthly = float(values.mean())
        monthly_std = float(values.std(ddof=1)) if n_months > 1 else math.nan
        t_stat = (
            mean_monthly / (monthly_std / math.sqrt(n_months))
            if n_months > 1 and monthly_std > 0
            else math.nan
        )
        records.append(
            {
                "portfolio": portfolio,
                "n_months": n_months,
                "mean_monthly_return": mean_monthly,
                "annualized_mean_return": 12.0 * mean_monthly,
                "annualized_volatility": math.sqrt(12.0) * monthly_std,
                "sharpe_like": (
                    math.sqrt(12.0) * mean_monthly / monthly_std
                    if monthly_std > 0
                    else math.nan
                ),
                "t_stat_mean_return": t_stat,
                "average_constituents": count_means.get(portfolio, math.nan),
            }
        )
    order = {f"Q{number}": number for number in range(1, 11)} | {"Q1-Q10": 11}
    result = pd.DataFrame(records)
    result["sort_order"] = result["portfolio"].map(order)
    return result.sort_values("sort_order").drop(columns="sort_order")


def add_long_short(monthly: pd.DataFrame) -> pd.DataFrame:
    wide = monthly.pivot(index="next_month", columns="portfolio", values="return")
    if "Q1" not in wide or "Q10" not in wide:
        raise ValueError("Q1 and Q10 are required for the long-short portfolio.")
    spread = (wide["Q1"] - wide["Q10"]).rename("return").reset_index()
    spread["portfolio"] = "Q1-Q10"
    return pd.concat([monthly, spread], ignore_index=True)


def save_decile_line(summary: pd.DataFrame, column: str, ylabel: str, path, title: str) -> None:
    deciles = summary.loc[summary["portfolio"].str.match(r"^Q\d+$")].copy()
    deciles["decile"] = deciles["portfolio"].str[1:].astype(int)
    deciles = deciles.sort_values("decile")
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(deciles["decile"], deciles[column], marker="o", linewidth=2)
    axis.set_xticks(range(1, 11), [f"Q{i}" for i in range(1, 11)])
    axis.set_xlabel("IVOL decile (low to high)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_decile_figure_data(
    summary_equal: pd.DataFrame, summary_value: pd.DataFrame
) -> pd.DataFrame:
    """Return one reviewed row per IVOL decile for the two-panel figure."""

    def select(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
        deciles = summary.loc[summary["portfolio"].str.fullmatch(r"Q\d+")].copy()
        deciles["decile"] = deciles["portfolio"].str[1:].astype(int)
        deciles = deciles.sort_values("decile")
        expected = list(range(1, 11))
        actual = deciles["decile"].tolist()
        if actual != expected:
            raise ValueError(
                f"Expected exactly one row for deciles 1-10; found {actual} for {prefix}."
            )
        return deciles[
            ["decile", "annualized_mean_return", "annualized_volatility"]
        ].rename(
            columns={
                "annualized_mean_return": f"{prefix}_annualized_mean_return",
                "annualized_volatility": f"{prefix}_annualized_volatility",
            }
        )

    equal = select(summary_equal, "equal_weight")
    value = select(summary_value, "value_weight")
    return equal.merge(value, on="decile", how="inner", validate="one_to_one")


def save_two_panel_decile_figure(
    chart_data: pd.DataFrame,
    png_path,
    formation_start: str,
    formation_end: str,
    return_start: str,
    return_end: str,
) -> None:
    """Save a paper-style return/volatility figure as a high-resolution PNG."""
    palette = {"Equal-weight": "#0072B2", "Value-weight": "#D55E00"}
    styles = {
        "Equal-weight": {"linestyle": "-", "marker": "o", "markerfacecolor": "#0072B2"},
        "Value-weight": {"linestyle": "--", "marker": "s", "markerfacecolor": "white"},
    }
    series = {
        "Equal-weight": "equal_weight",
        "Value-weight": "value_weight",
    }
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True)
        metric_specs = [
            ("annualized_mean_return", "Annualized Mean Return"),
            ("annualized_volatility", "Annualized Volatility"),
        ]
        handles = []
        labels = []
        for axis, (metric, ylabel) in zip(axes, metric_specs):
            for label, prefix in series.items():
                line, = axis.plot(
                    chart_data["decile"],
                    chart_data[f"{prefix}_{metric}"],
                    color=palette[label],
                    linewidth=2.0,
                    markersize=5.5,
                    markeredgewidth=1.3,
                    markeredgecolor=palette[label],
                    label=label,
                    **styles[label],
                )
                if axis is axes[0]:
                    handles.append(line)
                    labels.append(label)
            axis.set_xticks(range(1, 11))
            axis.set_xlabel("IVOL Signal Decile (1 = Low, 10 = High)")
            axis.set_ylabel(ylabel)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
            axis.set_axisbelow(True)
        axes[0].axhline(0.0, color="#4B5563", linewidth=0.9, zorder=0)
        axes[1].set_ylim(bottom=0.0)

        fig.suptitle(
            f"IVOL Decile Portfolio Performance ({return_start} to {return_end})",
            fontsize=15,
            fontweight="bold",
            y=0.97,
        )
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=2,
            frameon=False,
        )
        caption = (
            "Figure 1. IVOL portfolio performance by decile. The left panel shows annualized "
            "arithmetic mean returns; the right panel shows annualized volatility.\n"
            "Portfolios use month-t IVOL and month-(t+1) returns. Q1 is low IVOL; Q10 is high "
            "IVOL. EW = equal-weight; VW = formation-cap weight.\n"
            f"Formation months: {formation_start} to {formation_end}; return months: "
            f"{return_start} to {return_end}. Mean = 12 × monthly mean; volatility = √12 × "
            "monthly sample SD. Missing returns are excluded, not replaced with zero."
        )
        fig.text(
            0.035,
            0.02,
            caption,
            ha="left",
            va="bottom",
            fontsize=8.0,
            linespacing=1.35,
        )
        fig.tight_layout(rect=(0.025, 0.16, 0.985, 0.87), w_pad=2.5)
        png_path = ensure_parent(png_path)
        fig.savefig(png_path, dpi=300, facecolor="white")
        plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    output_dir = ensure_parent(f"{args.output_dir}/.sentinel").parent
    frame = pd.read_csv(args.input, compression="gzip")
    if frame.empty:
        raise RuntimeError("Decile assignment input is empty.")
    formation_start = str(frame["formation_month"].min())
    formation_end = str(frame["formation_month"].max())
    return_start = str(frame["next_month"].min())
    return_end = str(frame["next_month"].max())
    frame["next_month_return"] = pd.to_numeric(
        frame["next_month_return"], errors="coerce"
    )
    frame["formation_market_cap"] = pd.to_numeric(
        frame["formation_market_cap"], errors="coerce"
    )
    usable = frame.dropna(subset=["decile", "next_month", "next_month_return"]).copy()
    usable["decile"] = usable["decile"].astype(int)
    usable["portfolio"] = "Q" + usable["decile"].astype(str)

    equal = (
        usable.groupby(["next_month", "portfolio"], as_index=False)
        .agg(return_value=("next_month_return", "mean"), constituents=("PERMNO", "size"))
        .rename(columns={"return_value": "return"})
    )
    equal_with_spread = add_long_short(equal[["next_month", "portfolio", "return"]])
    count_means = equal.groupby("portfolio")["constituents"].mean().to_dict()
    summary_equal = portfolio_summary(equal_with_spread, count_means)

    value_eligible = usable.loc[usable["formation_market_cap"].gt(0)].copy()
    value_eligible["weighted_return"] = (
        value_eligible["next_month_return"] * value_eligible["formation_market_cap"]
    )
    value = (
        value_eligible.groupby(["next_month", "portfolio"], as_index=False)
        .agg(
            weighted_return=("weighted_return", "sum"),
            total_weight=("formation_market_cap", "sum"),
            constituents=("PERMNO", "size"),
        )
    )
    value["return"] = value["weighted_return"] / value["total_weight"]
    value_with_spread = add_long_short(value[["next_month", "portfolio", "return"]])
    value_count_means = value.groupby("portfolio")["constituents"].mean().to_dict()
    summary_value = portfolio_summary(value_with_spread, value_count_means)

    chart_data = build_decile_figure_data(summary_equal, summary_value)

    equal_with_spread.sort_values(["next_month", "portfolio"]).to_csv(
        output_dir / "monthly_returns_equal_weight.csv", index=False
    )
    value_with_spread.sort_values(["next_month", "portfolio"]).to_csv(
        output_dir / "monthly_returns_value_weight.csv", index=False
    )
    summary_equal.to_csv(output_dir / "summary_equal_weight.csv", index=False)
    summary_value.to_csv(output_dir / "summary_value_weight.csv", index=False)
    chart_data.to_csv(output_dir / "decile_figure_data.csv", index=False)

    save_decile_line(
        summary_equal,
        "annualized_mean_return",
        "Annualized mean return",
        output_dir / "decile_annualized_mean_return.png",
        f"IVOL deciles: annualized mean return ({return_start} to {return_end}, equal-weight)",
    )
    save_decile_line(
        summary_equal,
        "annualized_volatility",
        "Annualized volatility",
        output_dir / "decile_annualized_volatility.png",
        f"IVOL deciles: annualized volatility ({return_start} to {return_end}, equal-weight)",
    )
    save_two_panel_decile_figure(
        chart_data,
        output_dir / "decile_return_volatility_two_panel.png",
        formation_start,
        formation_end,
        return_start,
        return_end,
    )

    cumulative_source = equal_with_spread.loc[
        equal_with_spread["portfolio"].isin(["Q1", "Q10", "Q1-Q10"])
    ].copy()
    cumulative_source["gross"] = 1.0 + cumulative_source["return"]
    cumulative_source["cumulative"] = (
        cumulative_source.sort_values("next_month")
        .groupby("portfolio")["gross"]
        .cumprod()
        - 1.0
    )
    fig, axis = plt.subplots(figsize=(8, 4.8))
    palette = {"Q1": "#1f77b4", "Q10": "#d55e00", "Q1-Q10": "#7a5195"}
    line_styles = {"Q1": "-", "Q10": "--", "Q1-Q10": "-"}
    for portfolio in ["Q1", "Q10", "Q1-Q10"]:
        group = cumulative_source.loc[cumulative_source["portfolio"].eq(portfolio)]
        axis.plot(
            group["next_month"],
            group["cumulative"],
            label=portfolio,
            linewidth=2,
            color=palette[portfolio],
            linestyle=line_styles[portfolio],
        )
    axis.set_xlabel("Return month")
    axis.set_ylabel("Cumulative return")
    axis.set_title(
        f"Cumulative returns ({return_start} to {return_end}, equal-weight)"
    )
    months = sorted(cumulative_source["next_month"].unique())
    tick_positions = np.linspace(
        0, len(months) - 1, num=min(7, len(months)), dtype=int
    ).tolist()
    tick_positions = sorted(set(tick_positions))
    axis.set_xticks(tick_positions, [months[index] for index in tick_positions])
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_q1_q10_spread.png", dpi=160)
    plt.close(fig)

    audit = {
        "assignment_rows": int(len(frame)),
        "rows_with_next_month_return": int(len(usable)),
        "missing_next_month_return": int(frame["next_month_return"].isna().sum()),
        "value_weight_eligible_rows": int(len(value_eligible)),
        "return_months": sorted(usable["next_month"].unique().tolist()),
        "formation_months": [formation_start, formation_end],
        "return_period": [return_start, return_end],
        "long_short_direction": "Q1 low IVOL minus Q10 high IVOL",
        "annualization": {
            "mean": "12 * arithmetic mean monthly return",
            "volatility": "sqrt(12) * sample SD of monthly returns",
        },
        "inference_warning": (
            "The reported t-statistic uses an unadjusted IID standard error; use a "
            "HAC/Newey-West specification for final time-series inference."
        ),
        "two_panel_figure": "decile_return_volatility_two_panel.png",
    }
    write_json(audit, args.log)
    print(f"Wrote tables and figures to {output_dir}")


if __name__ == "__main__":
    main()
