from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from ivol_core import assign_deciles, write_json


REQUIRED_COLUMNS = [
    "PERMNO",
    "formation_month",
    "holding_month",
    "ivol_daily",
    "ivol_annualized",
    "n_daily_obs",
    "regression_rank",
    "ivol_decile",
    "ivol_rank_pct",
    "low_ivol_score",
]
ALLOWED_COLUMNS = {
    *REQUIRED_COLUMNS,
    "formation_date",
    "EXCHCD",
    "TICKER",
    "COMNAM",
}
FORBIDDEN_TOKENS = (
    "return",
    "ret_",
    "_ret",
    "mthret",
    "dlyret",
    "price",
    "_prc",
    "market_cap",
    "mthcap",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Validate a combination-ready IVOL file.")
    result.add_argument("--input", required=True)
    result.add_argument("--log", required=True)
    result.add_argument("--expected-rows", type=int)
    result.add_argument("--expected-months", type=int)
    return result


def validate(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    expected_months: int | None = None,
) -> dict[str, int | float | str]:
    checks: dict[str, bool] = {}
    checks["required_columns"] = not set(REQUIRED_COLUMNS).difference(frame.columns)
    forbidden = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    ]
    checks["no_outcome_like_columns"] = not forbidden
    unexpected = sorted(set(frame.columns).difference(ALLOWED_COLUMNS))
    checks["no_unexpected_columns"] = not unexpected
    if not checks["required_columns"]:
        missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
        raise ValueError(f"Missing required columns: {missing}")
    checks["nonempty"] = len(frame) > 0
    checks["unique_stock_month_key"] = not frame.duplicated(
        ["PERMNO", "formation_month"]
    ).any()
    checks["no_required_missing"] = not frame[REQUIRED_COLUMNS].isna().any().any()
    numeric_columns = [
        "ivol_daily",
        "ivol_annualized",
        "n_daily_obs",
        "regression_rank",
        "ivol_decile",
        "ivol_rank_pct",
        "low_ivol_score",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    checks["finite_numeric_values"] = bool(np.isfinite(numeric.to_numpy(float)).all())
    expected_holding = (
        pd.PeriodIndex(frame["formation_month"].astype(str), freq="M") + 1
    ).astype(str)
    checks["holding_is_next_month"] = frame["holding_month"].astype(str).eq(
        expected_holding
    ).all()
    checks["minimum_observations"] = pd.to_numeric(
        frame["n_daily_obs"], errors="coerce"
    ).ge(15).all()
    checks["full_regression_rank"] = pd.to_numeric(
        frame["regression_rank"], errors="coerce"
    ).eq(4).all()
    checks["nonnegative_ivol"] = pd.to_numeric(
        frame["ivol_daily"], errors="coerce"
    ).ge(0).all()
    checks["annualization"] = np.allclose(
        pd.to_numeric(frame["ivol_annualized"], errors="coerce"),
        pd.to_numeric(frame["ivol_daily"], errors="coerce") * math.sqrt(252.0),
        rtol=1e-10,
        atol=1e-12,
    )
    checks["decile_range"] = pd.to_numeric(
        frame["ivol_decile"], errors="coerce"
    ).between(1, 10).all()
    checks["rank_range"] = pd.to_numeric(
        frame["ivol_rank_pct"], errors="coerce"
    ).between(0, 1).all()
    checks["score_range"] = pd.to_numeric(
        frame["low_ivol_score"], errors="coerce"
    ).between(-1, 1).all()
    checks["score_matches_rank"] = np.allclose(
        numeric["low_ivol_score"],
        1.0 - 2.0 * numeric["ivol_rank_pct"],
        rtol=1e-12,
        atol=1e-12,
    )
    expected_decile = assign_deciles(frame)
    checks["decile_matches_ivol"] = numeric["ivol_decile"].eq(
        expected_decile.astype("int64")
    ).all()

    def expected_rank(values: pd.Series) -> pd.Series:
        if len(values) == 1:
            return pd.Series([0.5], index=values.index, dtype=float)
        return (values.rank(method="average", ascending=True) - 1.0) / (len(values) - 1.0)

    expected_rank_pct = frame.groupby("formation_month", sort=True, observed=True)[
        "ivol_daily"
    ].transform(expected_rank)
    checks["rank_matches_ivol"] = np.allclose(
        numeric["ivol_rank_pct"], expected_rank_pct, rtol=1e-12, atol=1e-12
    )
    monthly_deciles = frame.groupby("formation_month", observed=True)[
        "ivol_decile"
    ].agg(lambda values: set(pd.to_numeric(values, errors="coerce").dropna().astype(int)))
    checks["every_month_has_all_deciles"] = monthly_deciles.map(
        lambda values: values == set(range(1, 11))
    ).all()
    if expected_rows is not None:
        checks["expected_rows"] = len(frame) == expected_rows
    if expected_months is not None:
        checks["expected_months"] = frame["formation_month"].nunique() == expected_months

    failed = [name for name, passed in checks.items() if not passed]
    audit: dict[str, int | float | str] = {
        "rows": int(len(frame)),
        "formation_months": int(frame["formation_month"].nunique()),
        "unique_permnos": int(frame["PERMNO"].nunique()),
        "duplicate_keys": int(frame.duplicated(["PERMNO", "formation_month"]).sum()),
        "forbidden_columns": ",".join(forbidden),
        "unexpected_columns": ",".join(unexpected),
        "checks_total": int(len(checks)),
        "checks_passed": int(sum(checks.values())),
        "failed_checks": ",".join(failed),
    }
    if failed:
        raise ValueError(f"IVOL signal validation failed: {failed}")
    return audit


def main() -> None:
    args = parser().parse_args()
    frame = pd.read_csv(args.input, compression="infer", low_memory=False)
    audit = validate(
        frame,
        expected_rows=args.expected_rows,
        expected_months=args.expected_months,
    )
    write_json(audit, args.log)
    print(
        f"Validated {audit['rows']:,} IVOL rows; "
        f"{audit['checks_passed']}/{audit['checks_total']} checks passed"
    )


if __name__ == "__main__":
    main()
