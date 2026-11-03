from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description="Independently validate CNN OOS decile outputs.")
    result.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "cnn_oos_deciles_2001_2019",
    )
    return result


def newey_west_t(values: pd.Series, lag: int = 4) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    n_obs = len(clean)
    demeaned = clean - clean.mean()
    long_run_variance = float(demeaned @ demeaned / n_obs)
    for offset in range(1, min(lag, n_obs - 1) + 1):
        weight = 1.0 - offset / (lag + 1.0)
        covariance = float(demeaned[offset:] @ demeaned[:-offset] / n_obs)
        long_run_variance += 2.0 * weight * covariance
    return float(clean.mean() / math.sqrt(long_run_variance / n_obs))


def validate(output_dir: Path) -> dict[str, Any]:
    assignments = pd.read_csv(
        output_dir / "cnn_assignments_with_outcomes.csv.gz",
        compression="gzip",
        low_memory=False,
    )
    saved_monthly = pd.read_csv(output_dir / "cnn_monthly_returns_equal.csv")
    saved_summary = pd.read_csv(output_dir / "cnn_summary_equal.csv")

    expected_holding = (
        pd.PeriodIndex(assignments["formation_month"], freq="M") + 1
    ).astype(str)
    decile_sizes = assignments.groupby(
        ["formation_month", "cnn_decile"], observed=True
    ).size().unstack("cnn_decile")
    score_means = assignments.groupby(
        ["formation_month", "cnn_decile"], observed=True
    )["cnn_prob_up"].mean().unstack("cnn_decile")
    score_ordered = bool(
        (score_means.diff(axis=1).iloc[:, 1:] >= -1e-15).all().all()
    )

    valid_price = assignments["formation_adj_prc"].gt(0) & assignments[
        "holding_adj_prc"
    ].gt(0)
    recomputed_return = (
        assignments.loc[valid_price, "holding_adj_prc"]
        / assignments.loc[valid_price, "formation_adj_prc"]
        - 1.0
    )
    observed_return = assignments.loc[valid_price, "next_month_return"]
    return_error = np.abs(recomputed_return.to_numpy() - observed_return.to_numpy())

    usable = assignments.dropna(
        subset=["holding_month", "cnn_decile", "next_month_return"]
    ).copy()
    usable["portfolio"] = "Q" + usable["cnn_decile"].astype(int).astype(str)
    recomputed_monthly = (
        usable.groupby(["holding_month", "portfolio"], as_index=False, observed=True)[
            "next_month_return"
        ]
        .mean()
        .rename(columns={"next_month_return": "return"})
    )
    wide = recomputed_monthly.pivot(
        index="holding_month", columns="portfolio", values="return"
    )
    spread = (wide["Q10"] - wide["Q1"]).rename("return").reset_index()
    spread["portfolio"] = "Q10-Q1"
    recomputed_monthly = pd.concat(
        [recomputed_monthly, spread], ignore_index=True
    ).sort_values(["holding_month", "portfolio"])
    comparison = recomputed_monthly.merge(
        saved_monthly,
        on=["holding_month", "portfolio"],
        how="outer",
        suffixes=("_check", "_saved"),
        indicator=True,
        validate="one_to_one",
    )
    monthly_difference = np.abs(
        comparison["return_check"] - comparison["return_saved"]
    )

    spread_saved = saved_summary.loc[saved_summary["portfolio"].eq("Q10-Q1")].iloc[0]
    checks = {
        "assignment_keys_unique": not assignments.duplicated(
            ["PERMNO", "formation_month"]
        ).any(),
        "holding_month_is_exactly_t_plus_1": bool(
            assignments["holding_month"].astype(str).eq(expected_holding).all()
        ),
        "cnn_probability_is_finite_and_in_unit_interval": bool(
            np.isfinite(assignments["cnn_prob_up"]).all()
            and assignments["cnn_prob_up"].between(0, 1).all()
        ),
        "all_months_have_deciles_1_through_10": decile_sizes.columns.tolist()
        == list(range(1, 11)),
        "monthly_decile_sizes_differ_by_at_most_one": bool(
            (decile_sizes.max(axis=1) - decile_sizes.min(axis=1)).le(1).all()
        ),
        "mean_cnn_score_is_ordered_across_deciles": score_ordered,
        "valid_price_rows_have_recomputed_return": bool(
            assignments.loc[valid_price, "next_month_return"].notna().all()
        ),
        "invalid_price_rows_have_missing_return": bool(
            assignments.loc[~valid_price, "next_month_return"].isna().all()
        ),
        "next_month_return_matches_exact_price_ratio": bool(
            len(return_error) == 0 or float(return_error.max()) < 1e-12
        ),
        "monthly_return_keys_match_saved_output": bool(
            comparison["_merge"].eq("both").all()
        ),
        "monthly_equal_weight_returns_reproduce": bool(
            monthly_difference.dropna().empty
            or float(monthly_difference.max()) < 1e-12
        ),
        "saved_spread_is_q10_minus_q1": bool(
            np.allclose(
                wide["Q10"].to_numpy() - wide["Q1"].to_numpy(),
                spread["return"].to_numpy(),
                rtol=0,
                atol=1e-15,
            )
        ),
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "assignment_rows": int(len(assignments)),
        "formation_months": int(assignments["formation_month"].nunique()),
        "formation_month_start": str(assignments["formation_month"].min()),
        "formation_month_end": str(assignments["formation_month"].max()),
        "duplicate_assignment_keys": int(
            assignments.duplicated(["PERMNO", "formation_month"]).sum()
        ),
        "rows_with_next_month_return": int(assignments["next_month_return"].notna().sum()),
        "rows_missing_next_month_return": int(assignments["next_month_return"].isna().sum()),
        "maximum_return_recalculation_error": (
            float(return_error.max()) if len(return_error) else 0.0
        ),
        "maximum_monthly_return_reproduction_error": (
            float(monthly_difference.max()) if monthly_difference.notna().any() else 0.0
        ),
        "equal_weight_q10_minus_q1": {
            "n_months": int(len(spread)),
            "mean_monthly_return": float(spread["return"].mean()),
            "annualized_mean_return": float(12.0 * spread["return"].mean()),
            "annualized_volatility": float(
                math.sqrt(12.0) * spread["return"].std(ddof=1)
            ),
            "newey_west_t_lag4_recomputed": newey_west_t(spread["return"], lag=4),
            "newey_west_t_lag4_saved": float(
                spread_saved["t_stat_mean_return_newey_west_lag4"]
            ),
        },
    }
    if result["status"] != "pass":
        failures = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"CNN OOS validation failed: {failures}")
    return result


def main() -> None:
    args = parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    result = validate(output_dir)
    destination = output_dir / "cnn_oos_deciles_validation.json"
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(f"CNN OOS validation: {result['status']} ({sum(result['checks'].values())}/{len(result['checks'])})")


if __name__ == "__main__":
    main()
