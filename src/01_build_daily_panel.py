from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd

from common import (
    FF3_FILE,
    UNIVERSE_RULES,
    apply_universe_filter,
    ensure_parent,
    find_daily_files,
    load_ff3,
    resolve_data_dir,
    write_json,
)


USECOLS = [
    "PERMNO",
    "DlyCalDt",
    "DlyRet",
    "DlyRetMissFlg",
    "DlyDelFlg",
    *UNIVERSE_RULES.keys(),
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build a filtered daily CRSP/FF3 regression panel."
    )
    result.add_argument("--data-dir")
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
        raise ValueError("Daily CRSP is not sorted by PERMNO/DlyCalDt; safe streaming dedupe is unavailable.")

    duplicate_mask = chunk.duplicated(["PERMNO", "DlyCalDt"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = chunk.loc[duplicate_mask].copy()
        duplicate_rows["_row_hash"] = pd.util.hash_pandas_object(
            duplicate_rows[USECOLS].astype("string"), index=False, categorize=True
        ).to_numpy()
        conflicts = duplicate_rows.groupby(["PERMNO", "DlyCalDt"])["_row_hash"].nunique()
        if conflicts.gt(1).any():
            raise ValueError("Daily CRSP has conflicting rows for the same PERMNO/date key.")

    removed = int(chunk.duplicated(["PERMNO", "DlyCalDt"], keep="first").sum())
    chunk = chunk.drop_duplicates(["PERMNO", "DlyCalDt"], keep="first").copy()
    key_frame = key_frame.loc[chunk.index]

    first_key = (int(key_frame.iloc[0, 0]), str(key_frame.iloc[0, 1]))
    if last_key is not None and first_key < last_key:
        raise ValueError("Daily CRSP chunk order moved backwards across a boundary.")
    if last_key is not None and first_key == last_key:
        first_signature = tuple(chunk.iloc[0][USECOLS].fillna("<MISSING>").astype(str))
        if first_signature != last_signature:
            raise ValueError("Daily CRSP has a conflicting duplicate across chunk boundaries.")
        chunk = chunk.iloc[1:].copy()
        key_frame = key_frame.iloc[1:].copy()
        removed += 1
    if chunk.empty:
        return chunk, last_key, last_signature, removed

    new_last_key = (int(key_frame.iloc[-1, 0]), str(key_frame.iloc[-1, 1]))
    new_last_signature = tuple(chunk.iloc[-1][USECOLS].fillna("<MISSING>").astype(str))
    return chunk, new_last_key, new_last_signature, removed


def main() -> None:
    args = parser().parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    formation_start = pd.Period(args.formation_start, freq="M")
    formation_end = pd.Period(args.formation_end, freq="M")
    start_date = formation_start.start_time.normalize()
    end_date = formation_end.end_time.normalize()

    ff3 = load_ff3(data_dir / FF3_FILE)
    ff3 = ff3.loc[ff3["date"].between(start_date, end_date)].copy()
    if ff3.empty:
        raise ValueError("No FF3 observations overlap the requested formation window.")

    output = ensure_parent(args.output)
    audit = {
        "formation_start": str(formation_start),
        "formation_end": str(formation_end),
        "universe_rules": {key: sorted(value) for key, value in UNIVERSE_RULES.items()},
        "rows_in_date_range": 0,
        "rows_after_universe_filter": 0,
        "rows_with_valid_return": 0,
        "rows_with_complete_ff3": 0,
        "rows_written": 0,
        "exact_duplicate_rows_removed": 0,
        "ff3_dates_in_window": int(len(ff3)),
        "source_files_read": [],
    }
    filter_stage_totals: dict[str, int] = {}
    wrote_header = False
    last_key: tuple[int, str] | None = None
    last_signature: tuple[str, ...] | None = None

    with gzip.open(output, "wt", encoding="utf-8", newline="") as sink:
        for path in find_daily_files(data_dir):
            file_had_rows = False
            for chunk in pd.read_csv(
                path,
                compression="gzip",
                usecols=USECOLS,
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

                chunk, stage_audit = apply_universe_filter(chunk)
                for key, value in stage_audit.items():
                    filter_stage_totals[key] = filter_stage_totals.get(key, 0) + value
                audit["rows_after_universe_filter"] += int(len(chunk))

                chunk, last_key, last_signature, duplicates_removed = deduplicate_sorted_chunk(
                    chunk, last_key, last_signature
                )
                audit["exact_duplicate_rows_removed"] += duplicates_removed

                chunk["daily_return"] = pd.to_numeric(chunk["DlyRet"], errors="coerce")
                chunk = chunk.loc[chunk["daily_return"].notna()].copy()
                audit["rows_with_valid_return"] += int(len(chunk))

                chunk = chunk.merge(ff3, on="date", how="left", validate="many_to_one")
                factor_columns = ["mkt_rf", "smb", "hml", "rf"]
                complete_factors = chunk[factor_columns].notna().all(axis=1)
                chunk = chunk.loc[complete_factors].copy()
                audit["rows_with_complete_ff3"] += int(len(chunk))
                chunk["excess_return"] = chunk["daily_return"] - chunk["rf"]
                chunk["formation_month"] = chunk["date"].dt.to_period("M").astype(str)

                output_columns = [
                    "PERMNO",
                    "date",
                    "formation_month",
                    "excess_return",
                    "mkt_rf",
                    "smb",
                    "hml",
                    "rf",
                    "PrimaryExch",
                    "DlyRetMissFlg",
                    "DlyDelFlg",
                ]
                chunk[output_columns].to_csv(sink, index=False, header=not wrote_header)
                wrote_header = True
                audit["rows_written"] += int(len(chunk))
            if file_had_rows:
                audit["source_files_read"].append(path.name)
                last_key = None
                last_signature = None

    if not wrote_header:
        raise RuntimeError("No daily panel rows were written.")
    audit["filter_stage_totals"] = filter_stage_totals
    write_json(audit, args.log)
    print(f"Wrote {audit['rows_written']:,} daily rows to {output}")


if __name__ == "__main__":
    main()
