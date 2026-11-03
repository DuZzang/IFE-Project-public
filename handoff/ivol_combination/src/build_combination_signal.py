from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ivol_core import (
    assign_deciles,
    ensure_parent,
    load_professor_stock_data,
    official_formation_universe,
    write_json,
)


IVOL_COLUMNS = [
    "PERMNO",
    "formation_month",
    "ivol_daily",
    "ivol_annualized",
    "n_daily_obs",
    "regression_rank",
]
FORBIDDEN_FEATURE_TOKENS = ("return", "ret_", "price", "adj_prc", "market_cap")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Export a leakage-safe, combination-ready monthly IVOL signal."
    )
    result.add_argument("--professor-csv", required=True)
    result.add_argument("--ivol", required=True)
    result.add_argument("--formation-start", required=True, help="YYYY-MM")
    result.add_argument("--formation-end", required=True, help="YYYY-MM")
    result.add_argument("--output", required=True)
    result.add_argument("--log", required=True)
    return result


def _month_rank_percentile(values: pd.Series) -> pd.Series:
    n_rows = len(values)
    if n_rows == 1:
        return pd.Series([0.5], index=values.index, dtype=float)
    ranks = values.rank(method="average", ascending=True)
    return (ranks - 1.0) / (n_rows - 1.0)


def validate_ivol(ivol: pd.DataFrame, min_obs: int = 15) -> pd.DataFrame:
    missing = sorted(set(IVOL_COLUMNS).difference(ivol.columns))
    if missing:
        raise ValueError(f"IVOL file is missing required columns: {missing}")
    forbidden = [
        column
        for column in ivol.columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"IVOL feature input contains outcome-like columns: {forbidden}")

    result = ivol[IVOL_COLUMNS].copy()
    result["PERMNO"] = pd.to_numeric(result["PERMNO"], errors="raise").astype("int64")
    result["formation_month"] = result["formation_month"].astype(str)
    for column in [
        "ivol_daily",
        "ivol_annualized",
        "n_daily_obs",
        "regression_rank",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result.duplicated(["PERMNO", "formation_month"]).any():
        raise ValueError("IVOL file contains duplicate PERMNO/month keys.")
    if result[IVOL_COLUMNS].isna().any().any():
        raise ValueError("IVOL file contains missing required values.")
    if result["ivol_daily"].lt(0).any():
        raise ValueError("IVOL cannot be negative.")
    if result["n_daily_obs"].lt(min_obs).any():
        raise ValueError(f"IVOL contains rows with fewer than {min_obs} daily observations.")
    if result["regression_rank"].ne(4).any():
        raise ValueError("IVOL contains a regression whose design rank is not four.")
    expected_annualized = result["ivol_daily"] * math.sqrt(252.0)
    if not np.allclose(
        result["ivol_annualized"], expected_annualized, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("ivol_annualized is inconsistent with ivol_daily * sqrt(252).")
    return result


def build_signal(
    professor: pd.DataFrame,
    ivol: pd.DataFrame,
    *,
    formation_start: str,
    formation_end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = str(pd.Period(formation_start, freq="M"))
    end = str(pd.Period(formation_end, freq="M"))
    if pd.Period(end, freq="M") < pd.Period(start, freq="M"):
        raise ValueError("formation-end cannot precede formation-start")
    formation = official_formation_universe(professor, start, end).rename(
        columns={"date": "formation_date"}
    )
    ivol = validate_ivol(ivol)
    ivol = ivol.loc[ivol["formation_month"].between(start, end)].copy()
    if ivol.empty:
        raise ValueError("No IVOL rows overlap the requested formation period.")

    key_check = ivol[["PERMNO", "formation_month"]].merge(
        formation[["PERMNO", "formation_month"]],
        on=["PERMNO", "formation_month"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    outside = int(key_check["_merge"].ne("both").sum())
    if outside:
        raise ValueError(f"IVOL contains {outside:,} keys outside the professor universe.")

    signal = formation[
        ["PERMNO", "formation_date", "formation_month", "EXCHCD", "TICKER", "COMNAM"]
    ].merge(
        ivol,
        on=["PERMNO", "formation_month"],
        how="left",
        validate="one_to_one",
    )
    missing_ivol = int(signal["ivol_daily"].isna().sum())
    signal = signal.dropna(subset=["ivol_daily"]).copy()
    # The left join temporarily promotes these integer diagnostics to float when
    # some professor-universe rows have no valid IVOL. Restore their semantic
    # types after those rows are removed so the handoff CSV is unambiguous.
    signal["n_daily_obs"] = signal["n_daily_obs"].astype("int64")
    signal["regression_rank"] = signal["regression_rank"].astype("int64")
    signal["holding_month"] = (
        pd.PeriodIndex(signal["formation_month"], freq="M") + 1
    ).astype(str)
    signal["ivol_decile"] = assign_deciles(signal)
    signal["ivol_decile"] = signal["ivol_decile"].astype("int64")
    signal["ivol_rank_pct"] = signal.groupby(
        "formation_month", sort=True, observed=True
    )["ivol_daily"].transform(_month_rank_percentile)
    # Combination scores commonly use higher=better. This signed orientation is
    # +1 for the lowest IVOL and -1 for the highest IVOL in each month.
    signal["low_ivol_score"] = 1.0 - 2.0 * signal["ivol_rank_pct"]
    signal = signal.sort_values(["formation_month", "PERMNO"]).reset_index(drop=True)

    if signal.duplicated(["PERMNO", "formation_month"]).any():
        raise AssertionError("Signal output contains duplicate stock-month keys.")
    if signal["holding_month"].ne(
        (pd.PeriodIndex(signal["formation_month"], freq="M") + 1).astype(str)
    ).any():
        raise AssertionError("holding_month is not exactly formation_month + 1.")
    if not signal["ivol_rank_pct"].between(0, 1).all():
        raise AssertionError("ivol_rank_pct falls outside [0, 1].")
    if not signal["low_ivol_score"].between(-1, 1).all():
        raise AssertionError("low_ivol_score falls outside [-1, 1].")

    monthly = signal.groupby("formation_month", observed=True).agg(
        rows=("PERMNO", "size"),
        minimum_decile=("ivol_decile", "min"),
        maximum_decile=("ivol_decile", "max"),
    )
    audit: dict[str, Any] = {
        "formation_start": start,
        "formation_end": end,
        "holding_start": str(pd.Period(start, freq="M") + 1),
        "holding_end": str(pd.Period(end, freq="M") + 1),
        "professor_formation_rows": int(len(formation)),
        "valid_ivol_rows": int(len(signal)),
        "formation_rows_without_ivol": missing_ivol,
        "coverage_rate": float(len(signal) / len(formation)),
        "unique_permnos": int(signal["PERMNO"].nunique()),
        "formation_months": int(signal["formation_month"].nunique()),
        "duplicate_keys": int(signal.duplicated(["PERMNO", "formation_month"]).sum()),
        "ivol_decile_definition": "1=lowest IVOL, 10=highest IVOL; ties by PERMNO",
        "ivol_rank_pct_definition": "monthly average rank scaled to [0,1]; higher=riskier",
        "low_ivol_score_definition": "1 - 2*ivol_rank_pct; higher=lower IVOL",
        "missing_policy": "Never fill missing IVOL with zero",
        "timing_rule": "formation-month t signal is eligible only for holding month t+1",
        "monthly_decile_range_failures": int(
            ((monthly["minimum_decile"] != 1) | (monthly["maximum_decile"] != 10)).sum()
        ),
    }
    columns = [
        "PERMNO",
        "formation_date",
        "formation_month",
        "holding_month",
        "EXCHCD",
        "TICKER",
        "COMNAM",
        "ivol_daily",
        "ivol_annualized",
        "n_daily_obs",
        "regression_rank",
        "ivol_decile",
        "ivol_rank_pct",
        "low_ivol_score",
    ]
    return signal[columns], audit


def main() -> None:
    args = parser().parse_args()
    professor, professor_audit = load_professor_stock_data(args.professor_csv)
    ivol = pd.read_csv(args.ivol, compression="infer", low_memory=False)
    signal, signal_audit = build_signal(
        professor,
        ivol,
        formation_start=args.formation_start,
        formation_end=args.formation_end,
    )
    output = ensure_parent(args.output)
    signal.to_csv(output, index=False, compression="infer")
    write_json(
        {
            "professor_source": professor_audit,
            "ivol_source_file": Path(args.ivol).name,
            **signal_audit,
        },
        args.log,
    )
    print(f"Wrote {len(signal):,} combination-ready IVOL rows to {output}")


if __name__ == "__main__":
    main()
