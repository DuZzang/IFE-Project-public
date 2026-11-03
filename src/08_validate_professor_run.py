from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Independently validate the completed professor-universe run."
    )
    result.add_argument("--assignments", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--report-figure", required=True)
    result.add_argument("--professor-csv", required=True)
    result.add_argument("--log", required=True)
    return result


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def independently_summarize(
    frame: pd.DataFrame, value_weighted: bool
) -> pd.DataFrame:
    usable = frame.dropna(subset=["next_month_return", "decile"]).copy()
    usable["decile"] = usable["decile"].astype(int)
    if value_weighted:
        usable = usable.loc[usable["formation_market_cap"].gt(0)].copy()
        usable["weighted"] = (
            usable["next_month_return"] * usable["formation_market_cap"]
        )
        monthly = (
            usable.groupby(["next_month", "decile"], as_index=False, observed=True)
            .agg(weighted=("weighted", "sum"), cap=("formation_market_cap", "sum"))
        )
        monthly["return"] = monthly["weighted"] / monthly["cap"]
    else:
        monthly = (
            usable.groupby(["next_month", "decile"], as_index=False, observed=True)[
                "next_month_return"
            ]
            .mean()
            .rename(columns={"next_month_return": "return"})
        )
    return (
        monthly.groupby("decile", as_index=False, observed=True)
        .agg(
            n_months=("return", "size"),
            mean_monthly=("return", "mean"),
            monthly_sd=("return", lambda values: values.std(ddof=1)),
        )
        .assign(
            annualized_mean_return=lambda data: 12.0 * data["mean_monthly"],
            annualized_volatility=lambda data: math.sqrt(12.0) * data["monthly_sd"],
        )
    )


def main() -> None:
    args = parser().parse_args()
    assignments = pd.read_csv(args.assignments, compression="gzip", low_memory=False)
    assignments["next_month_return"] = pd.to_numeric(
        assignments["next_month_return"], errors="coerce"
    )
    assignments["formation_market_cap"] = pd.to_numeric(
        assignments["formation_market_cap"], errors="coerce"
    )
    output_dir = Path(args.output_dir)
    chart_data = pd.read_csv(output_dir / "decile_figure_data.csv")

    checks: dict[str, bool] = {}
    checks["assignment_key_unique"] = not assignments.duplicated(
        ["PERMNO", "formation_month"]
    ).any()
    expected_next = (
        pd.PeriodIndex(assignments["formation_month"], freq="M") + 1
    ).astype(str)
    checks["next_month_is_exactly_t_plus_1"] = np.array_equal(
        expected_next.to_numpy(), assignments["next_month"].astype(str).to_numpy()
    )
    checks["formation_exchange_is_1_2_3"] = bool(
        assignments["EXCHCD"].isin([1, 2, 3]).all()
    )
    checks["deciles_are_1_through_10"] = sorted(
        assignments["decile"].dropna().astype(int).unique().tolist()
    ) == list(range(1, 11))

    decile_sizes = assignments.groupby(
        ["formation_month", "decile"], observed=True
    ).size().unstack("decile")
    checks["monthly_decile_sizes_differ_by_at_most_one"] = bool(
        (decile_sizes.max(axis=1) - decile_sizes.min(axis=1)).le(1).all()
    )
    bounds = assignments.groupby(
        ["formation_month", "decile"], observed=True
    )["ivol_daily"].agg(["min", "max"])
    ordered = True
    for _, month_bounds in bounds.groupby(level=0, observed=True):
        month_bounds = month_bounds.reset_index(level=0, drop=True).sort_index()
        ordered &= bool(
            (
                month_bounds["max"].iloc[:-1].to_numpy()
                <= month_bounds["min"].iloc[1:].to_numpy()
            ).all()
        )
    checks["ivol_is_ordered_across_deciles"] = ordered

    valid_prices = assignments[
        ["formation_adj_prc", "next_month_adj_prc"]
    ].gt(0).all(axis=1)
    recomputed_returns = (
        assignments.loc[valid_prices, "next_month_adj_prc"]
        / assignments.loc[valid_prices, "formation_adj_prc"]
        - 1.0
    )
    checks["adj_prc_return_recomputes_exactly"] = bool(
        np.allclose(
            recomputed_returns,
            assignments.loc[valid_prices, "next_month_return"],
            rtol=1e-12,
            atol=1e-12,
        )
    )
    checks["missing_price_pairs_remain_missing_returns"] = bool(
        assignments.loc[~valid_prices, "next_month_return"].isna().all()
    )

    professor_keys = pd.read_csv(
        args.professor_csv,
        usecols=["PERMNO", "date", "EXCHCD"],
        low_memory=False,
    )
    professor_keys["formation_month"] = pd.to_datetime(
        professor_keys["date"], errors="raise"
    ).dt.to_period("M").astype(str)
    professor_keys = professor_keys.loc[
        professor_keys["formation_month"].between("2000-01", "2020-11")
        & professor_keys["EXCHCD"].isin([1, 2, 3]),
        ["PERMNO", "formation_month"],
    ]
    membership = assignments[["PERMNO", "formation_month"]].merge(
        professor_keys,
        on=["PERMNO", "formation_month"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    checks["every_assignment_is_in_professor_formation_universe"] = bool(
        membership["_merge"].eq("both").all()
    )

    equal = independently_summarize(assignments, value_weighted=False)
    value = independently_summarize(assignments, value_weighted=True)
    merged = chart_data.merge(
        equal[
            ["decile", "n_months", "annualized_mean_return", "annualized_volatility"]
        ].rename(
            columns={
                "n_months": "check_equal_n_months",
                "annualized_mean_return": "check_equal_return",
                "annualized_volatility": "check_equal_volatility",
            }
        ),
        on="decile",
        validate="one_to_one",
    ).merge(
        value[
            ["decile", "n_months", "annualized_mean_return", "annualized_volatility"]
        ].rename(
            columns={
                "n_months": "check_value_n_months",
                "annualized_mean_return": "check_value_return",
                "annualized_volatility": "check_value_volatility",
            }
        ),
        on="decile",
        validate="one_to_one",
    )
    checks["chart_has_exactly_deciles_1_through_10"] = (
        merged["decile"].tolist() == list(range(1, 11))
    )
    checks["all_deciles_have_251_equal_weight_months"] = bool(
        merged["check_equal_n_months"].eq(251).all()
    )
    checks["all_deciles_have_251_value_weight_months"] = bool(
        merged["check_value_n_months"].eq(251).all()
    )
    checks["chart_equal_weight_values_match_independent_recompute"] = bool(
        np.allclose(
            merged[
                [
                    "equal_weight_annualized_mean_return",
                    "equal_weight_annualized_volatility",
                ]
            ],
            merged[["check_equal_return", "check_equal_volatility"]],
            rtol=1e-12,
            atol=1e-12,
        )
    )
    checks["chart_value_weight_values_match_independent_recompute"] = bool(
        np.allclose(
            merged[
                [
                    "value_weight_annualized_mean_return",
                    "value_weight_annualized_volatility",
                ]
            ],
            merged[["check_value_return", "check_value_volatility"]],
            rtol=1e-12,
            atol=1e-12,
        )
    )

    output_figure = output_dir / "decile_return_volatility_two_panel.png"
    report_figure = Path(args.report_figure)
    with Image.open(output_figure) as image:
        image_size = list(image.size)
        checks["figure_is_valid_3600_by_2040_png"] = (
            image.format == "PNG" and image.size == (3600, 2040)
        )
    checks["output_and_report_figure_hashes_match"] = (
        sha256(output_figure) == sha256(report_figure)
    )

    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise AssertionError(f"Professor-run validation failed: {failed}")

    observed_returns = assignments["next_month_return"].dropna()
    payload = {
        "status": "passed",
        "checks": checks,
        "assignment_rows": int(len(assignments)),
        "professor_formation_rows": int(len(professor_keys)),
        "assignment_coverage_of_professor_formation": float(
            len(assignments) / len(professor_keys)
        ),
        "missing_next_month_returns": int(
            assignments["next_month_return"].isna().sum()
        ),
        "missing_next_month_return_rate": float(
            assignments["next_month_return"].isna().mean()
        ),
        "observed_return_min": float(observed_returns.min()),
        "observed_return_max": float(observed_returns.max()),
        "figure_dimensions_pixels": image_size,
        "figure_sha256": sha256(output_figure),
        "professor_csv_sha256": sha256(args.professor_csv),
        "independent_decile_summary": {
            "equal_weight": equal.to_dict(orient="records"),
            "value_weight": value.to_dict(orient="records"),
        },
    }
    write_json(payload, args.log)
    print(f"All {len(checks)} independent validation checks passed.")


if __name__ == "__main__":
    main()
