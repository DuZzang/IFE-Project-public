from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from ivol_core import ensure_parent, ivol_from_moments, write_json


MOMENT_COLUMNS = [
    "n",
    "y",
    "m",
    "s",
    "h",
    "mm",
    "ms",
    "mh",
    "ss",
    "sh",
    "hh",
    "my",
    "sy",
    "hy",
    "yy",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Estimate stock-month FF3 residual volatility from a daily panel."
    )
    result.add_argument("--input", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--log", required=True)
    result.add_argument("--min-obs", type=int, default=15)
    result.add_argument("--chunksize", type=int, default=500_000)
    result.add_argument(
        "--allow-prevalidated-without-return-flag",
        action="store_true",
        help=(
            "Explicitly allow an already-filtered daily panel that omits "
            "DlyRetMissFlg. The standard pipeline never needs this override."
        ),
    )
    return result


def chunk_moments(chunk: pd.DataFrame) -> pd.DataFrame:
    required = {"PERMNO", "formation_month", "excess_return", "mkt_rf", "smb", "hml"}
    missing = sorted(required.difference(chunk.columns))
    if missing:
        raise ValueError(f"Daily panel is missing columns: {missing}")
    numeric_columns = ["excess_return", "mkt_rf", "smb", "hml"]
    for column in numeric_columns:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    chunk = chunk.dropna(
        subset=["PERMNO", "formation_month", *numeric_columns]
    ).copy()
    y = chunk["excess_return"].to_numpy(float)
    m = chunk["mkt_rf"].to_numpy(float)
    s = chunk["smb"].to_numpy(float)
    h = chunk["hml"].to_numpy(float)
    enriched = pd.DataFrame(
        {
            "PERMNO": chunk["PERMNO"].to_numpy(),
            "formation_month": chunk["formation_month"].to_numpy(),
            "n": np.ones(len(chunk), dtype=np.int64),
            "y": y,
            "m": m,
            "s": s,
            "h": h,
            "mm": m * m,
            "ms": m * s,
            "mh": m * h,
            "ss": s * s,
            "sh": s * h,
            "hh": h * h,
            "my": m * y,
            "sy": s * y,
            "hy": h * y,
            "yy": y * y,
        }
    )
    return enriched.groupby(["PERMNO", "formation_month"], as_index=False)[
        MOMENT_COLUMNS
    ].sum()


def row_to_ivol(row: object) -> tuple[float, int]:
    xtx = np.array(
        [
            [row.n, row.m, row.s, row.h],
            [row.m, row.mm, row.ms, row.mh],
            [row.s, row.ms, row.ss, row.sh],
            [row.h, row.mh, row.sh, row.hh],
        ],
        dtype=float,
    )
    xty = np.array([row.y, row.my, row.sy, row.hy], dtype=float)
    return ivol_from_moments(xtx, xty, float(row.yy), int(row.n))


def compute_ivol_from_moments(
    moments: pd.DataFrame, min_obs: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    if min_obs < 5:
        raise ValueError("min-obs must be at least 5 for a four-parameter regression")
    records: list[dict[str, object]] = []
    insufficient = 0
    singular = 0
    for row in moments.itertuples(index=False):
        if int(row.n) < min_obs:
            insufficient += 1
            continue
        ivol, rank = row_to_ivol(row)
        if math.isnan(ivol):
            singular += 1
            continue
        records.append(
            {
                "PERMNO": int(row.PERMNO),
                "formation_month": str(row.formation_month),
                "ivol_daily": ivol,
                "ivol_annualized": ivol * math.sqrt(252.0),
                "n_daily_obs": int(row.n),
                "regression_rank": rank,
            }
        )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise RuntimeError("No valid stock-month IVOL estimates were produced.")
    if result.duplicated(["PERMNO", "formation_month"]).any():
        raise AssertionError("IVOL output contains duplicate stock-month keys.")
    result = result.sort_values(["formation_month", "PERMNO"]).reset_index(drop=True)
    return result, {
        "rejected_insufficient_observations": insufficient,
        "rejected_singular_regressions": singular,
    }


def main() -> None:
    args = parser().parse_args()
    if args.chunksize <= 0:
        raise ValueError("chunksize must be positive")
    input_header = pd.read_csv(args.input, compression="infer", nrows=0).columns
    return_flag_present = "DlyRetMissFlg" in input_header
    if not return_flag_present and not args.allow_prevalidated_without_return_flag:
        raise ValueError(
            "Daily panel is missing DlyRetMissFlg. Use the standard panel builder, "
            "or pass --allow-prevalidated-without-return-flag only after independently "
            "verifying that unavailable CRSP returns were removed."
        )
    partials: list[pd.DataFrame] = []
    input_rows = 0
    rows_excluded_nonavailable_return_flag = 0
    for chunk in pd.read_csv(
        args.input,
        compression="infer",
        chunksize=args.chunksize,
        keep_default_na=False,
        na_values=[""],
        dtype={"DlyRetMissFlg": "string"},
        low_memory=False,
    ):
        input_rows += len(chunk)
        if "DlyRetMissFlg" in chunk:
            available = chunk["DlyRetMissFlg"].astype("string").str.strip().eq("NA")
            rows_excluded_nonavailable_return_flag += int((~available).sum())
            chunk = chunk.loc[available].copy()
        partials.append(chunk_moments(chunk))
    if not partials:
        raise RuntimeError("Daily panel is empty.")

    moments = pd.concat(partials, ignore_index=True)
    moments = moments.groupby(["PERMNO", "formation_month"], as_index=False)[
        MOMENT_COLUMNS
    ].sum()
    result, rejection_audit = compute_ivol_from_moments(moments, args.min_obs)
    output = ensure_parent(args.output)
    result.to_csv(output, index=False, compression="infer")
    write_json(
        {
            "input_daily_rows": int(input_rows),
            "return_missing_flag_present": return_flag_present,
            "return_flag_policy": (
                "Require DlyRetMissFlg=NA; omission requires an explicit prevalidated override"
            ),
            "prevalidated_without_return_flag_override": bool(
                args.allow_prevalidated_without_return_flag
            ),
            "rows_excluded_nonavailable_return_flag": int(
                rows_excluded_nonavailable_return_flag
            ),
            "stock_month_candidates": int(len(moments)),
            "minimum_daily_observations": int(args.min_obs),
            **rejection_audit,
            "valid_stock_months": int(len(result)),
            "ivol_definition": "sample standard deviation of OLS residuals (ddof=1)",
            "regressors": ["intercept", "Mkt-RF", "SMB", "HML"],
            "annualization": "ivol_daily * sqrt(252)",
        },
        args.log,
    )
    print(f"Wrote {len(result):,} stock-month IVOL estimates to {output}")


if __name__ == "__main__":
    main()
