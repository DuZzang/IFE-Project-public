from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from performance_metrics import add_long_short, summarize_portfolios
from shared_contract import (
    build_adj_price_outcomes,
    load_formation_caps,
    load_professor_prices,
    write_json,
)
from validate_signal import validate as validate_combination_signal


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Evaluate IVOL deciles with the shared t-to-t+1 outcome contract."
    )
    result.add_argument("--professor-csv", required=True)
    result.add_argument("--signal", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument(
        "--crsp-monthly",
        help="Optional CRSP monthly file for formation-cap value-weight robustness.",
    )
    result.add_argument("--chunksize", type=int, default=500_000)
    return result


def load_signal(path: str | Path) -> pd.DataFrame:
    signal = pd.read_csv(path, compression="infer", low_memory=False)
    required = {
        "PERMNO",
        "formation_month",
        "holding_month",
        "ivol_daily",
        "ivol_decile",
        "low_ivol_score",
    }
    missing = sorted(required.difference(signal.columns))
    if missing:
        raise ValueError(f"IVOL signal is missing columns: {missing}")
    signal["PERMNO"] = pd.to_numeric(signal["PERMNO"], errors="raise").astype("int64")
    signal["formation_month"] = signal["formation_month"].astype(str)
    signal["holding_month"] = signal["holding_month"].astype(str)
    signal["ivol_decile"] = pd.to_numeric(
        signal["ivol_decile"], errors="raise"
    ).astype("int64")
    if signal.duplicated(["PERMNO", "formation_month"]).any():
        raise ValueError("IVOL signal has duplicate stock-month keys")
    expected_holding = (
        pd.PeriodIndex(signal["formation_month"], freq="M") + 1
    ).astype(str)
    if not signal["holding_month"].eq(expected_holding).all():
        raise ValueError("IVOL signal violates formation t to holding t+1 timing")
    if not signal["ivol_decile"].between(1, 10).all():
        raise ValueError("ivol_decile is outside 1 through 10")
    expected_months = len(
        pd.period_range(
            signal["formation_month"].min(), signal["formation_month"].max(), freq="M"
        )
    )
    validate_combination_signal(signal, expected_months=expected_months)
    return signal


def build_monthly_portfolios(
    assignments: pd.DataFrame,
    *,
    weight_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"PERMNO", "holding_month", "ivol_decile", "next_month_return"}
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise ValueError(f"Assignment table is missing columns: {missing}")
    usable = assignments.dropna(
        subset=["holding_month", "ivol_decile", "next_month_return"]
    ).copy()
    usable["portfolio"] = "Q" + usable["ivol_decile"].astype(int).astype(str)
    if weight_column is None:
        monthly = (
            usable.groupby(["holding_month", "portfolio"], as_index=False, observed=True)
            .agg(return_value=("next_month_return", "mean"), constituents=("PERMNO", "size"))
            .rename(columns={"return_value": "return"})
        )
        weighting = "equal_weight"
    else:
        if weight_column not in usable:
            raise ValueError(f"Assignment table is missing weight column: {weight_column}")
        usable[weight_column] = pd.to_numeric(usable[weight_column], errors="coerce")
        usable = usable.loc[usable[weight_column].gt(0)].copy()
        usable["weighted_return"] = usable["next_month_return"] * usable[weight_column]
        monthly = usable.groupby(
            ["holding_month", "portfolio"], as_index=False, observed=True
        ).agg(
            weighted_return=("weighted_return", "sum"),
            total_weight=(weight_column, "sum"),
            constituents=("PERMNO", "size"),
        )
        monthly["return"] = monthly["weighted_return"] / monthly["total_weight"]
        monthly = monthly[["holding_month", "portfolio", "return", "constituents"]]
        weighting = "value_weight"
    expected = {f"Q{i}" for i in range(1, 11)}
    observed = set(monthly["portfolio"].unique())
    if expected.difference(observed):
        raise ValueError(f"Missing IVOL deciles: {sorted(expected.difference(observed))}")
    with_spread = add_long_short(
        monthly[["holding_month", "portfolio", "return"]],
        long_label="Q1",
        short_label="Q10",
        spread_label="Q1-Q10",
    )
    audit = {
        "weighting": weighting,
        "assignment_rows": int(len(assignments)),
        "rows_with_usable_return_and_weight": int(len(usable)),
        "missing_next_month_return": int(assignments["next_month_return"].isna().sum()),
        "return_months": int(with_spread["holding_month"].nunique()),
        "return_month_start": str(with_spread["holding_month"].min()),
        "return_month_end": str(with_spread["holding_month"].max()),
        "average_constituents": {
            str(key): float(value)
            for key, value in monthly.groupby("portfolio", observed=True)[
                "constituents"
            ].mean().items()
        },
    }
    return with_spread.sort_values(["holding_month", "portfolio"]), audit


def main() -> None:
    args = parser().parse_args()
    if args.chunksize <= 0:
        raise ValueError("chunksize must be positive")
    signal = load_signal(args.signal)
    professor, professor_audit = load_professor_prices(args.professor_csv)
    formation_start = str(signal["formation_month"].min())
    formation_end = str(signal["formation_month"].max())
    outcomes, outcome_audit = build_adj_price_outcomes(
        professor,
        formation_start=formation_start,
        formation_end=formation_end,
    )
    outcome_columns = [
        "PERMNO",
        "formation_month",
        "holding_month",
        "formation_adj_prc",
        "holding_adj_prc",
        "next_month_return",
    ]
    assignments = signal.merge(
        outcomes[outcome_columns],
        on=["PERMNO", "formation_month", "holding_month"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    missing_formation_keys = int(assignments["_merge"].ne("both").sum())
    if missing_formation_keys:
        raise ValueError(
            f"Signal contains {missing_formation_keys:,} keys outside the professor universe"
        )
    assignments = assignments.drop(columns=["_merge"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(
        output_dir / "ivol_assignments_with_outcomes.csv.gz",
        index=False,
        compression="gzip",
    )

    equal_monthly, equal_audit = build_monthly_portfolios(assignments)
    equal_summary = summarize_portfolios(equal_monthly)
    equal_monthly.to_csv(output_dir / "ivol_monthly_returns_equal.csv", index=False)
    equal_summary.to_csv(output_dir / "ivol_summary_equal.csv", index=False)

    value_audit: dict[str, Any] | None = None
    cap_audit: dict[str, Any] | None = None
    if args.crsp_monthly:
        caps, cap_audit = load_formation_caps(
            args.crsp_monthly,
            signal[["PERMNO", "formation_month"]],
            chunksize=args.chunksize,
        )
        assignments_with_caps = assignments.merge(
            caps,
            on=["PERMNO", "formation_month"],
            how="left",
            validate="one_to_one",
        )
        value_monthly, value_audit = build_monthly_portfolios(
            assignments_with_caps,
            weight_column="formation_market_cap",
        )
        value_summary = summarize_portfolios(value_monthly)
        value_monthly.to_csv(output_dir / "ivol_monthly_returns_value.csv", index=False)
        value_summary.to_csv(output_dir / "ivol_summary_value.csv", index=False)

    audit = {
        "signal_file": Path(args.signal).name,
        "formation_period": [formation_start, formation_end],
        "timing": "formation month t; holding and outcome month t+1",
        "primary_weighting": "equal weight",
        "primary_spread": "Q1 low IVOL minus Q10 high IVOL",
        "performance_convention": {
            "annualized_mean": "12 * monthly arithmetic mean",
            "annualized_volatility": "sqrt(12) * monthly sample SD",
            "newey_west_lag": 4,
        },
        "professor": professor_audit,
        "outcomes": outcome_audit,
        "equal_weight": equal_audit,
        "value_weight": value_audit,
        "formation_caps": cap_audit,
    }
    write_json(audit, output_dir / "ivol_evaluation_audit.json")
    print(f"Wrote IVOL evaluation to {output_dir}")


if __name__ == "__main__":
    main()
