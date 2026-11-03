from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    MONTHLY_FILE,
    UNIVERSE_RULES,
    apply_universe_filter,
    assign_deciles,
    ensure_parent,
    resolve_data_dir,
    write_json,
)


USECOLS = [
    "PERMNO",
    "MthCalDt",
    "MthRet",
    "MthCap",
    "MthPrevCap",
    "MthRetFlg",
    "MthDelFlg",
    *UNIVERSE_RULES.keys(),
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Assign monthly IVOL deciles and join next-month CRSP returns."
    )
    result.add_argument("--data-dir")
    result.add_argument("--ivol", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--log", required=True)
    result.add_argument("--chunksize", type=int, default=300_000)
    return result


def load_monthly_window(
    path, start_month: pd.Period, end_month: pd.Period, chunksize: int
) -> pd.DataFrame:
    pieces = []
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=USECOLS,
        chunksize=chunksize,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    ):
        dates = pd.to_datetime(chunk["MthCalDt"], errors="coerce")
        months = dates.dt.to_period("M")
        mask = months.between(start_month, end_month)
        if not mask.any():
            continue
        selected = chunk.loc[mask].copy()
        selected["month"] = months.loc[mask].astype(str)
        pieces.append(selected)
    if not pieces:
        raise RuntimeError("No monthly CRSP rows overlap the requested window.")
    result = pd.concat(pieces, ignore_index=True)
    result["PERMNO"] = pd.to_numeric(result["PERMNO"], errors="coerce").astype("Int64")
    for column in ["MthRet", "MthCap", "MthPrevCap"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def deduplicate_monthly(
    frame: pd.DataFrame,
    comparison_columns: list[str],
    label: str,
) -> tuple[pd.DataFrame, int]:
    duplicate_mask = frame.duplicated(["PERMNO", "month"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = frame.loc[duplicate_mask].copy()
        duplicate_rows["_row_hash"] = pd.util.hash_pandas_object(
            duplicate_rows[comparison_columns].astype("string"),
            index=False,
            categorize=True,
        ).to_numpy()
        conflicts = duplicate_rows.groupby(["PERMNO", "month"])["_row_hash"].nunique()
        if conflicts.gt(1).any():
            raise ValueError(
                f"Monthly CRSP has conflicting {label} rows for the same PERMNO/month key."
            )
    removed = int(
        frame.duplicated(["PERMNO", "month"], keep="first").sum()
    )
    return frame.drop_duplicates(["PERMNO", "month"], keep="first").copy(), removed


def main() -> None:
    args = parser().parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    ivol = pd.read_csv(args.ivol, compression="gzip")
    ivol["PERMNO"] = pd.to_numeric(ivol["PERMNO"], errors="raise").astype("Int64")
    ivol["formation_month"] = ivol["formation_month"].astype(str)
    ivol["ivol_daily"] = pd.to_numeric(ivol["ivol_daily"], errors="coerce")
    ivol = ivol.dropna(subset=["PERMNO", "formation_month", "ivol_daily"]).copy()

    if ivol.duplicated(["PERMNO", "formation_month"]).any():
        raise ValueError("IVOL input has duplicate PERMNO/formation_month keys.")

    formation_start = pd.Period(ivol["formation_month"].min(), freq="M")
    formation_end = pd.Period(ivol["formation_month"].max(), freq="M")
    return_end = formation_end + 1
    monthly_raw = load_monthly_window(
        data_dir / MONTHLY_FILE, formation_start, return_end, args.chunksize
    )
    formation_source, _ = apply_universe_filter(monthly_raw)
    formation_columns = [column for column in formation_source.columns if column != "month"]
    formation_source, formation_duplicates_removed = deduplicate_monthly(
        formation_source,
        comparison_columns=formation_columns,
        label="formation-universe",
    )

    outcome_columns = ["MthRet", "MthPrevCap", "MthCap", "MthDelFlg", "MthRetFlg"]
    return_source, outcome_duplicates_removed = deduplicate_monthly(
        monthly_raw,
        comparison_columns=outcome_columns,
        label="return-outcome",
    )

    formation = formation_source.loc[
        formation_source["month"].between(str(formation_start), str(formation_end)),
        ["PERMNO", "month", "MthCap"],
    ].rename(columns={"month": "formation_month", "MthCap": "formation_market_cap"})
    returns = return_source.loc[
        return_source["month"].between(str(formation_start + 1), str(return_end)),
        ["PERMNO", "month", "MthRet", "MthPrevCap", "MthRetFlg", "MthDelFlg"],
    ].rename(
        columns={
            "month": "next_month",
            "MthRet": "next_month_return",
            "MthPrevCap": "return_row_prev_cap",
        }
    )

    assignments = ivol.merge(
        formation,
        on=["PERMNO", "formation_month"],
        how="left",
        validate="one_to_one",
    )
    assignments["decile"] = assign_deciles(assignments)
    assignments["next_month"] = (
        pd.PeriodIndex(assignments["formation_month"], freq="M") + 1
    ).astype(str)
    assignments = assignments.merge(
        returns,
        on=["PERMNO", "next_month"],
        how="left",
        validate="one_to_one",
    )

    comparable_cap = (
        assignments["formation_market_cap"].gt(0)
        & assignments["return_row_prev_cap"].gt(0)
    )
    cap_close = np.isclose(
        assignments.loc[comparable_cap, "formation_market_cap"],
        assignments.loc[comparable_cap, "return_row_prev_cap"],
        rtol=1e-8,
        atol=1e-10,
    )
    cap_mismatch_count = int((~cap_close).sum())

    output_columns = [
        "PERMNO",
        "formation_month",
        "ivol_daily",
        "ivol_annualized",
        "n_daily_obs",
        "decile",
        "next_month",
        "next_month_return",
        "formation_market_cap",
        "return_row_prev_cap",
        "MthRetFlg",
        "MthDelFlg",
    ]
    output = ensure_parent(args.output)
    assignments[output_columns].sort_values(
        ["formation_month", "decile", "PERMNO"]
    ).to_csv(output, index=False, compression="gzip")

    decile_counts = (
        assignments.groupby(["formation_month", "decile"], observed=True)
        .size()
        .groupby("decile")
        .agg(["min", "max", "mean"])
        .reset_index()
        .to_dict(orient="records")
    )
    audit = {
        "formation_start": str(formation_start),
        "formation_end": str(formation_end),
        "return_start": str(formation_start + 1),
        "return_end": str(return_end),
        "ivol_rows": int(len(ivol)),
        "monthly_raw_rows_in_window": int(len(monthly_raw)),
        "monthly_formation_rows_after_filter": int(len(formation_source)),
        "formation_exact_duplicate_rows_removed": formation_duplicates_removed,
        "return_outcome_duplicate_rows_collapsed": outcome_duplicates_removed,
        "return_filter_rule": "No outcome-month eligibility refilter; join MthRet by PERMNO and next_month to avoid survivorship bias.",
        "missing_formation_market_cap": int(
            assignments["formation_market_cap"].isna().sum()
        ),
        "missing_next_month_return": int(
            assignments["next_month_return"].isna().sum()
        ),
        "cap_values_compared": int(comparable_cap.sum()),
        "formation_cap_vs_next_row_prev_cap_mismatches": cap_mismatch_count,
        "tie_rule": "sort by IVOL ascending, then PERMNO ascending; assign floor(rank*10/n)+1",
        "decile_counts": decile_counts,
    }
    write_json(audit, args.log)
    print(f"Wrote {len(assignments):,} decile assignments to {output}")


if __name__ == "__main__":
    main()
