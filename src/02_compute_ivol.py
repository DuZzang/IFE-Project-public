from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from common import ensure_parent, ivol_from_moments, write_json


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
    return result


def chunk_moments(chunk: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = ["excess_return", "mkt_rf", "smb", "hml"]
    for column in numeric_columns:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    chunk = chunk.dropna(subset=["PERMNO", "formation_month", *numeric_columns]).copy()
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


def row_to_ivol(row: pd.Series) -> tuple[float, int]:
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


def main() -> None:
    args = parser().parse_args()
    partials = []
    input_rows = 0
    for chunk in pd.read_csv(
        args.input,
        compression="gzip",
        chunksize=args.chunksize,
        dtype={"DlyRetMissFlg": "string", "DlyDelFlg": "string"},
        low_memory=False,
    ):
        input_rows += len(chunk)
        partials.append(chunk_moments(chunk))
    if not partials:
        raise RuntimeError("Daily panel is empty.")

    moments = pd.concat(partials, ignore_index=True)
    moments = moments.groupby(["PERMNO", "formation_month"], as_index=False)[
        MOMENT_COLUMNS
    ].sum()

    records = []
    insufficient = 0
    singular = 0
    for row in moments.itertuples(index=False):
        if int(row.n) < args.min_obs:
            insufficient += 1
            continue
        ivol, rank = row_to_ivol(row)
        if math.isnan(ivol):
            singular += 1
            continue
        records.append(
            {
                "PERMNO": int(row.PERMNO),
                "formation_month": row.formation_month,
                "ivol_daily": ivol,
                "ivol_annualized": ivol * math.sqrt(252.0),
                "n_daily_obs": int(row.n),
                "regression_rank": rank,
            }
        )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise RuntimeError("No valid stock-month IVOL estimates were produced.")
    result = result.sort_values(["formation_month", "PERMNO"])
    output = ensure_parent(args.output)
    result.to_csv(output, index=False, compression="gzip")
    audit = {
        "input_daily_rows": int(input_rows),
        "stock_month_candidates": int(len(moments)),
        "minimum_daily_observations": int(args.min_obs),
        "rejected_insufficient_observations": int(insufficient),
        "rejected_singular_regressions": int(singular),
        "valid_stock_months": int(len(result)),
        "ivol_definition": "sample standard deviation of OLS residuals (ddof=1)",
        "regressors": ["intercept", "Mkt-RF", "SMB", "HML"],
    }
    write_json(audit, args.log)
    print(f"Wrote {len(result):,} stock-month IVOL estimates to {output}")


if __name__ == "__main__":
    main()
