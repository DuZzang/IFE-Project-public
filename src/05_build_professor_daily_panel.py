from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd

from common import FF3_FILE, ensure_parent, find_daily_files, load_ff3, resolve_data_dir, write_json
from professor_universe import load_professor_stock_data, official_formation_universe


DAILY_USECOLS = [
    "PERMNO",
    "DlyCalDt",
    "DlyRet",
    "DlyRetMissFlg",
    "DlyDelFlg",
    "ConditionalType",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build a daily FF3 panel for the professor CSV universe."
    )
    result.add_argument("--data-dir")
    result.add_argument("--professor-csv", required=True)
    result.add_argument("--formation-start", required=True, help="YYYY-MM")
    result.add_argument("--formation-end", required=True, help="YYYY-MM")
    result.add_argument("--output", required=True)
    result.add_argument("--log", required=True)
    result.add_argument("--chunksize", type=int, default=500_000)
    return result


def deduplicate_sorted_chunk(
    chunk: pd.DataFrame,
    last_key: tuple[int, str] | None,
    last_signature: tuple[str, ...] | None,
) -> tuple[pd.DataFrame, tuple[int, str] | None, tuple[str, ...] | None, int]:
    """Safely collapse exact duplicates in a PERMNO/date-sorted CRSP stream."""
    if chunk.empty:
        return chunk, last_key, last_signature, 0
    key_frame = pd.DataFrame(
        {
            "PERMNO": pd.to_numeric(chunk["PERMNO"], errors="raise").astype("int64"),
            "DlyCalDt": chunk["DlyCalDt"].astype(str),
        },
        index=chunk.index,
    )
    key_index = pd.MultiIndex.from_frame(key_frame)
    if not key_index.is_monotonic_increasing:
        raise ValueError(
            "Daily CRSP is not sorted by PERMNO/DlyCalDt; streaming dedupe is unsafe."
        )

    duplicate_mask = chunk.duplicated(["PERMNO", "DlyCalDt"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = chunk.loc[duplicate_mask].copy()
        duplicate_rows["_row_hash"] = pd.util.hash_pandas_object(
            duplicate_rows[DAILY_USECOLS].astype("string"),
            index=False,
            categorize=True,
        ).to_numpy()
        conflicts = duplicate_rows.groupby(["PERMNO", "DlyCalDt"], observed=True)[
            "_row_hash"
        ].nunique()
        if conflicts.gt(1).any():
            raise ValueError("Daily CRSP has conflicting duplicate PERMNO/date rows.")

    removed = int(chunk.duplicated(["PERMNO", "DlyCalDt"], keep="first").sum())
    chunk = chunk.drop_duplicates(["PERMNO", "DlyCalDt"], keep="first").copy()
    key_frame = key_frame.loc[chunk.index]

    first_key = (int(key_frame.iloc[0, 0]), str(key_frame.iloc[0, 1]))
    if last_key is not None and first_key < last_key:
        raise ValueError("Daily CRSP order moved backwards across a chunk boundary.")
    if last_key is not None and first_key == last_key:
        first_signature = tuple(
            chunk.iloc[0][DAILY_USECOLS].fillna("<MISSING>").astype(str)
        )
        if first_signature != last_signature:
            raise ValueError("Daily CRSP has a conflicting duplicate at a chunk boundary.")
        chunk = chunk.iloc[1:].copy()
        key_frame = key_frame.iloc[1:].copy()
        removed += 1
    if chunk.empty:
        return chunk, last_key, last_signature, removed

    new_last_key = (int(key_frame.iloc[-1, 0]), str(key_frame.iloc[-1, 1]))
    new_last_signature = tuple(
        chunk.iloc[-1][DAILY_USECOLS].fillna("<MISSING>").astype(str)
    )
    return chunk, new_last_key, new_last_signature, removed


def main() -> None:
    args = parser().parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    formation_start = pd.Period(args.formation_start, freq="M")
    formation_end = pd.Period(args.formation_end, freq="M")
    start_date = formation_start.start_time.normalize()
    end_date = formation_end.end_time.normalize()

    professor, professor_audit = load_professor_stock_data(args.professor_csv)
    membership = official_formation_universe(
        professor, formation_start=formation_start, formation_end=formation_end
    )[["PERMNO", "formation_month"]].copy()
    membership["PERMNO"] = membership["PERMNO"].astype("int64")
    membership["formation_month"] = membership["formation_month"].astype(str)

    ff3 = load_ff3(data_dir / FF3_FILE)
    ff3 = ff3.loc[ff3["date"].between(start_date, end_date)].copy()
    if ff3.empty:
        raise ValueError("No FF3 rows overlap the requested formation window.")

    output = ensure_parent(args.output)
    audit = {
        "formation_start": str(formation_start),
        "formation_end": str(formation_end),
        "professor_source": professor_audit,
        "professor_formation_keys": int(len(membership)),
        "daily_filter_rule": "Professor PERMNO/month membership AND ConditionalType=RW AND DlyRetMissFlg=NA",
        "excluded_entity_filters": [
            "USIncFlg",
            "IssuerType",
            "SecurityType",
            "SecuritySubType",
            "ShareType",
            "PrimaryExch",
        ],
        "rows_in_date_range": 0,
        "rows_after_conditional_type_rw": 0,
        "rows_after_professor_membership": 0,
        "rows_excluded_nonavailable_return_flag": 0,
        "rows_with_valid_return": 0,
        "rows_with_complete_ff3": 0,
        "rows_written": 0,
        "exact_duplicate_rows_removed": 0,
        "ff3_dates_in_window": int(len(ff3)),
        "source_files_read": [],
    }
    wrote_header = False

    with gzip.open(output, "wt", encoding="utf-8", newline="") as sink:
        for path in find_daily_files(data_dir):
            file_had_rows = False
            last_key: tuple[int, str] | None = None
            last_signature: tuple[str, ...] | None = None
            for chunk in pd.read_csv(
                path,
                compression="gzip",
                usecols=DAILY_USECOLS,
                chunksize=args.chunksize,
                keep_default_na=False,
                na_values=[""],
                dtype="string",
                low_memory=False,
            ):
                dates = pd.to_datetime(chunk["DlyCalDt"], errors="coerce")
                in_window = dates.between(start_date, end_date)
                if not in_window.any():
                    continue
                file_had_rows = True
                chunk = chunk.loc[in_window].copy()
                chunk["date"] = dates.loc[in_window]
                audit["rows_in_date_range"] += int(len(chunk))

                chunk, last_key, last_signature, removed = deduplicate_sorted_chunk(
                    chunk, last_key, last_signature
                )
                audit["exact_duplicate_rows_removed"] += removed
                if chunk.empty:
                    continue

                rw = chunk["ConditionalType"].astype("string").str.strip().eq("RW")
                chunk = chunk.loc[rw].copy()
                audit["rows_after_conditional_type_rw"] += int(len(chunk))
                if chunk.empty:
                    continue

                chunk["PERMNO"] = pd.to_numeric(
                    chunk["PERMNO"], errors="raise"
                ).astype("int64")
                chunk["formation_month"] = chunk["date"].dt.to_period("M").astype(str)
                chunk = chunk.merge(
                    membership,
                    on=["PERMNO", "formation_month"],
                    how="inner",
                    validate="many_to_one",
                )
                audit["rows_after_professor_membership"] += int(len(chunk))
                if chunk.empty:
                    continue

                chunk["daily_return"] = pd.to_numeric(chunk["DlyRet"], errors="coerce")
                chunk = chunk.loc[chunk["daily_return"].notna()].copy()
                # CRSP DlyRetMissFlg "NA" = return available; e.g. "MV" (missing corporate-action value) is excluded.
                available = chunk["DlyRetMissFlg"].astype("string").str.strip().eq("NA")
                audit["rows_excluded_nonavailable_return_flag"] += int((~available).sum())
                chunk = chunk.loc[available].copy()
                audit["rows_with_valid_return"] += int(len(chunk))

                chunk = chunk.merge(ff3, on="date", how="left", validate="many_to_one")
                factor_columns = ["mkt_rf", "smb", "hml", "rf"]
                complete_factors = chunk[factor_columns].notna().all(axis=1)
                chunk = chunk.loc[complete_factors].copy()
                audit["rows_with_complete_ff3"] += int(len(chunk))
                chunk["excess_return"] = chunk["daily_return"] - chunk["rf"]

                output_columns = [
                    "PERMNO",
                    "date",
                    "formation_month",
                    "excess_return",
                    "mkt_rf",
                    "smb",
                    "hml",
                    "rf",
                    "DlyRetMissFlg",
                    "DlyDelFlg",
                ]
                chunk[output_columns].to_csv(
                    sink, index=False, header=not wrote_header
                )
                wrote_header = True
                audit["rows_written"] += int(len(chunk))
            if file_had_rows:
                audit["source_files_read"].append(path.name)

    if not wrote_header:
        raise RuntimeError("No professor-universe daily panel rows were written.")
    write_json(audit, args.log)
    print(f"Wrote {audit['rows_written']:,} professor-universe daily rows to {output}")


if __name__ == "__main__":
    main()
