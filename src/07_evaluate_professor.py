from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from common import ensure_parent, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Evaluate professor-universe IVOL decile portfolios."
    )
    result.add_argument("--input", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--log", required=True)
    result.add_argument("--report-figure")
    return result


def newey_west_t_stat(values: pd.Series, lag: int = 4) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    n_obs = len(clean)
    if n_obs <= lag or n_obs < 2:
        return math.nan
    demeaned = clean - clean.mean()
    long_run_variance = float(demeaned @ demeaned) / n_obs
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        autocovariance = float(demeaned[offset:] @ demeaned[:-offset]) / n_obs
        long_run_variance += 2.0 * weight * autocovariance
    if long_run_variance <= 0:
        return math.nan
    return float(clean.mean() / math.sqrt(long_run_variance / n_obs))


def add_long_short(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.duplicated(["next_month", "portfolio"]).any():
        raise ValueError("Monthly portfolio input has duplicate month/portfolio keys.")
    wide = monthly.pivot(index="next_month", columns="portfolio", values="return")
    if "Q1" not in wide or "Q10" not in wide:
        raise ValueError("Q1 and Q10 are required for the long-short portfolio.")
    spread = (wide["Q1"] - wide["Q10"]).rename("return").reset_index()
    spread["portfolio"] = "Q1-Q10"
    return pd.concat([monthly, spread], ignore_index=True)


def portfolio_summary(
    monthly: pd.DataFrame, count_means: dict[str, float]
) -> pd.DataFrame:
    records = []
    for portfolio, values in monthly.groupby("portfolio", observed=True)["return"]:
        values = values.dropna()
        n_months = len(values)
        mean_monthly = float(values.mean())
        monthly_std = float(values.std(ddof=1)) if n_months > 1 else math.nan
        iid_t = (
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
                "t_stat_mean_return_iid": iid_t,
                "t_stat_mean_return_newey_west_lag4": newey_west_t_stat(values, lag=4),
                "average_constituents_with_return": count_means.get(
                    portfolio, math.nan
                ),
            }
        )
    order = {f"Q{number}": number for number in range(1, 11)} | {"Q1-Q10": 11}
    result = pd.DataFrame(records)
    result["sort_order"] = result["portfolio"].map(order)
    return result.sort_values("sort_order").drop(columns="sort_order").reset_index(drop=True)


def build_monthly_portfolios(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame = assignments.copy()
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
        usable.groupby(["next_month", "portfolio"], as_index=False, observed=True)
        .agg(return_value=("next_month_return", "mean"), constituents=("PERMNO", "size"))
        .rename(columns={"return_value": "return"})
    )
    count_means = equal.groupby("portfolio", observed=True)["constituents"].mean().to_dict()
    equal_with_spread = add_long_short(equal[["next_month", "portfolio", "return"]])

    value_eligible = usable.loc[usable["formation_market_cap"].gt(0)].copy()
    value_eligible["weighted_return"] = (
        value_eligible["next_month_return"] * value_eligible["formation_market_cap"]
    )
    value = (
        value_eligible.groupby(
            ["next_month", "portfolio"], as_index=False, observed=True
        )
        .agg(
            weighted_return=("weighted_return", "sum"),
            total_weight=("formation_market_cap", "sum"),
            constituents=("PERMNO", "size"),
        )
    )
    value["return"] = value["weighted_return"] / value["total_weight"]
    value_count_means = (
        value.groupby("portfolio", observed=True)["constituents"].mean().to_dict()
    )
    value_with_spread = add_long_short(value[["next_month", "portfolio", "return"]])

    expected_months = sorted(frame["next_month"].dropna().astype(str).unique().tolist())
    expected_portfolios = [f"Q{i}" for i in range(1, 11)]
    month_coverage: dict[str, dict[str, int]] = {}
    for label, monthly in [("equal_weight", equal), ("value_weight", value)]:
        counts = monthly.groupby("portfolio", observed=True)["next_month"].nunique()
        actual_portfolios = sorted(counts.index.tolist())
        if sorted(expected_portfolios) != actual_portfolios:
            raise ValueError(f"{label} is missing one or more IVOL decile portfolios.")
        if not counts.eq(len(expected_months)).all():
            raise ValueError(
                f"{label} does not have complete month coverage in every decile: "
                f"{counts.to_dict()}"
            )
        month_coverage[label] = {key: int(value) for key, value in counts.items()}

    details = {
        "assignment_rows": int(len(frame)),
        "rows_with_next_month_return": int(len(usable)),
        "missing_next_month_return": int(frame["next_month_return"].isna().sum()),
        "value_weight_eligible_rows": int(len(value_eligible)),
        "expected_return_months": int(len(expected_months)),
        "return_month_start": expected_months[0],
        "return_month_end": expected_months[-1],
        "month_coverage_by_decile": month_coverage,
        "equal_count_means": count_means,
        "value_count_means": value_count_means,
    }
    return equal_with_spread, value_with_spread, details


def build_figure_data(
    summary_equal: pd.DataFrame, summary_value: pd.DataFrame
) -> pd.DataFrame:
    def select(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
        deciles = summary.loc[summary["portfolio"].str.fullmatch(r"Q\d+")].copy()
        deciles["decile"] = deciles["portfolio"].str[1:].astype(int)
        deciles = deciles.sort_values("decile")
        if deciles["decile"].tolist() != list(range(1, 11)):
            raise ValueError(f"Expected exactly deciles 1-10 for {prefix}.")
        return deciles[
            [
                "decile",
                "n_months",
                "annualized_mean_return",
                "annualized_volatility",
                "average_constituents_with_return",
            ]
        ].rename(
            columns={
                column: f"{prefix}_{column}"
                for column in [
                    "n_months",
                    "annualized_mean_return",
                    "annualized_volatility",
                    "average_constituents_with_return",
                ]
            }
        )

    result = select(summary_equal, "equal_weight").merge(
        select(summary_value, "value_weight"),
        on="decile",
        how="inner",
        validate="one_to_one",
    )
    metric_columns = [
        column
        for column in result.columns
        if column.endswith("annualized_mean_return")
        or column.endswith("annualized_volatility")
    ]
    if not np.isfinite(result[metric_columns].to_numpy(float)).all():
        raise ValueError("Figure data contains a non-finite return or volatility.")
    return result


def save_two_panel_figure(
    chart_data: pd.DataFrame,
    path: str | Path,
    formation_start: str,
    formation_end: str,
    return_start: str,
    return_end: str,
    n_months: int,
) -> None:
    palette = {"EW (primary)": "#0072B2", "VW (CRSP cap)": "#D55E00"}
    styles = {
        "EW (primary)": {
            "linestyle": "-",
            "marker": "o",
            "markerfacecolor": "#0072B2",
        },
        "VW (CRSP cap)": {
            "linestyle": "--",
            "marker": "s",
            "markerfacecolor": "white",
        },
    }
    series = {"EW (primary)": "equal_weight", "VW (CRSP cap)": "value_weight"}
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
        specs = [
            ("annualized_mean_return", "Annualized Mean Return", "Realized Return"),
            ("annualized_volatility", "Annualized Volatility", "Realized Volatility"),
        ]
        handles = []
        labels = []
        for axis, (metric, ylabel, panel_title) in zip(axes, specs):
            for label, prefix in series.items():
                line, = axis.plot(
                    chart_data["decile"],
                    chart_data[f"{prefix}_{metric}"],
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
            axis.set_title(panel_title, fontsize=12, pad=9)
            axis.set_xticks(range(1, 11))
            axis.set_xlabel("IVOL Signal Decile (1 = Low, 10 = High)")
            axis.set_ylabel(ylabel)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
            axis.set_axisbelow(True)
        axes[0].axhline(0.0, color="#4B5563", linewidth=0.9, zorder=0)
        axes[1].set_ylim(bottom=0.0)

        fig.suptitle(
            "IVOL Decile Portfolio Performance - Professor CSV Universe",
            fontsize=15,
            fontweight="bold",
            y=0.975,
        )
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=2,
            frameon=False,
        )
        caption = (
            "Figure 1. Month-t IVOL deciles in the professor CSV universe (formation EXCHCD 1/2/3); "
            "month-(t+1) returns use exact adjacent-month ADJ_PRC changes.\n"
            "EW is the primary professor-file result. VW uses supplemental CRSP formation-month market cap. "
            "Missing next-month prices are excluded, never set to zero.\n"
            f"Formation: {formation_start} to {formation_end}; returns: {return_start} to {return_end} "
            f"({n_months} months). Mean = 12 x monthly arithmetic mean; volatility = sqrt(12) x monthly sample SD."
        )
        fig.text(0.035, 0.022, caption, ha="left", va="bottom", fontsize=8.1, linespacing=1.4)
        fig.tight_layout(rect=(0.025, 0.175, 0.985, 0.875), w_pad=2.6)
        output = ensure_parent(path)
        fig.savefig(output, dpi=300, facecolor="white")
        plt.close(fig)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parser().parse_args()
    output_dir = ensure_parent(Path(args.output_dir) / ".sentinel").parent
    frame = pd.read_csv(args.input, compression="gzip", low_memory=False)
    if frame.empty:
        raise RuntimeError("Professor-universe assignment input is empty.")

    formation_start = str(frame["formation_month"].min())
    formation_end = str(frame["formation_month"].max())
    return_start = str(frame["next_month"].min())
    return_end = str(frame["next_month"].max())
    equal, value, details = build_monthly_portfolios(frame)
    summary_equal = portfolio_summary(
        equal, details["equal_count_means"]
    )
    summary_value = portfolio_summary(
        value, details["value_count_means"]
    )
    chart_data = build_figure_data(summary_equal, summary_value)

    equal.sort_values(["next_month", "portfolio"]).to_csv(
        output_dir / "monthly_returns_equal_weight.csv", index=False
    )
    value.sort_values(["next_month", "portfolio"]).to_csv(
        output_dir / "monthly_returns_value_weight.csv", index=False
    )
    summary_equal.to_csv(output_dir / "summary_equal_weight.csv", index=False)
    summary_value.to_csv(output_dir / "summary_value_weight.csv", index=False)
    chart_data.to_csv(output_dir / "decile_figure_data.csv", index=False)

    figure_path = output_dir / "decile_return_volatility_two_panel.png"
    save_two_panel_figure(
        chart_data,
        figure_path,
        formation_start,
        formation_end,
        return_start,
        return_end,
        details["expected_return_months"],
    )
    report_hash = None
    if args.report_figure:
        report_path = ensure_parent(args.report_figure)
        shutil.copyfile(figure_path, report_path)
        report_hash = sha256(report_path)
        if report_hash != sha256(figure_path):
            raise AssertionError("Output and report figure copies do not match.")

    summary_records = {
        "equal_weight": summary_equal.to_dict(orient="records"),
        "value_weight": summary_value.to_dict(orient="records"),
    }
    audit = {
        **{key: value for key, value in details.items() if not key.endswith("count_means")},
        "formation_period": [formation_start, formation_end],
        "return_period": [return_start, return_end],
        "primary_result": "equal-weight",
        "value_weight_source": "CRSP MthCap at formation month",
        "return_source": "Professor CSV ADJ_PRC exact t to t+1 change",
        "long_short_direction": "Q1 low IVOL minus Q10 high IVOL",
        "annualization": {
            "mean": "12 * arithmetic mean monthly return",
            "volatility": "sqrt(12) * sample SD of monthly portfolio returns",
        },
        "figure": figure_path.name,
        "figure_sha256": sha256(figure_path),
        "report_figure_sha256": report_hash,
        "summary": summary_records,
    }
    write_json(audit, args.log)
    print(f"Wrote professor-universe tables and figure to {output_dir}")


if __name__ == "__main__":
    main()
