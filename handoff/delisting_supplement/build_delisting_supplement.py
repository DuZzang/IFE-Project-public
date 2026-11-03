"""Build one shared next-month return file that includes CRSP delisting returns.

The professor CSV has only ADJ_PRC, so a stock that delists (or otherwise has no
t+1 row) silently drops out of every strategy's portfolio return. This script
keeps the professor return as the default and uses CRSP CIZ ``MthRet`` (which
carries the delisting return in the delisting month) only where it is needed:

1. CRSP flags month t+1 as a delisting month and has a return -> CRSP MthRet
2. otherwise the professor ADJ_PRC(t+1)/ADJ_PRC(t)-1 exists     -> ADJ_PRC return
3. otherwise CRSP has a t+1 return                              -> CRSP MthRet
4. otherwise                                                    -> missing (never 0)

Output grain: one row per professor formation stock-month with EXCHCD 1/2/3.
Only aggregate counts are printed; no row-level values are written to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXCHANGE_CODES = {1, 2, 3}
DEFAULT_FORMATION_START = "2000-01"
DEFAULT_FORMATION_END = "2020-11"
CRSP_COLUMNS = ["PERMNO", "MthCalDt", "MthRet", "MthRetFlg", "MthDelFlg"]
NO_DELISTING_FLAG = "N"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("--professor-csv", type=Path, required=True)
    result.add_argument("--crsp-monthly", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--qa", type=Path, required=True)
    result.add_argument("--formation-start", default=DEFAULT_FORMATION_START)
    result.add_argument("--formation-end", default=DEFAULT_FORMATION_END)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_professor(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["PERMNO", "date", "EXCHCD", "ADJ_PRC"],
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    frame["PERMNO"] = pd.to_numeric(frame["PERMNO"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["PERMNO"].isna().any() or frame["date"].isna().any():
        raise ValueError("Professor CSV has rows with an invalid PERMNO or date.")
    frame["PERMNO"] = frame["PERMNO"].astype("int64")
    frame["EXCHCD"] = pd.to_numeric(frame["EXCHCD"], errors="coerce").astype("Int64")
    frame["ADJ_PRC"] = pd.to_numeric(frame["ADJ_PRC"], errors="coerce")
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    if frame.duplicated(["PERMNO", "month"]).any():
        raise ValueError("Professor CSV must have one row per PERMNO/month.")
    return frame


def professor_outcomes(professor: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Formation rows (EXCHCD 1/2/3 at t) with the exact adjacent-month ADJ_PRC return."""
    formation = professor.loc[
        professor["month"].between(start, end)
        & professor["EXCHCD"].isin(EXCHANGE_CODES),
        ["PERMNO", "month", "EXCHCD", "ADJ_PRC"],
    ].rename(columns={"month": "formation_month", "ADJ_PRC": "formation_adj_prc"})
    formation["holding_month"] = (
        pd.PeriodIndex(formation["formation_month"], freq="M") + 1
    ).astype(str)
    future = professor[["PERMNO", "month", "ADJ_PRC"]].rename(
        columns={"month": "holding_month", "ADJ_PRC": "holding_adj_prc"}
    )
    outcomes = formation.merge(
        future, on=["PERMNO", "holding_month"], how="left", validate="one_to_one"
    )
    valid = outcomes["formation_adj_prc"].gt(0) & outcomes["holding_adj_prc"].gt(0)
    outcomes["ret_adjprc"] = np.where(
        valid, outcomes["holding_adj_prc"] / outcomes["formation_adj_prc"] - 1.0, np.nan
    )
    return outcomes.drop(columns=["formation_adj_prc", "holding_adj_prc"])


def load_crsp_monthly(path: Path, first: str, last: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(
        path,
        usecols=CRSP_COLUMNS,
        dtype="string",
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    frame["holding_month"] = (
        pd.to_datetime(frame["MthCalDt"], errors="coerce").dt.to_period("M").astype("string")
    )
    frame = frame.loc[frame["holding_month"].between(first, last)].copy()
    frame["PERMNO"] = pd.to_numeric(frame["PERMNO"], errors="coerce")
    frame = frame.dropna(subset=["PERMNO", "holding_month"]).copy()
    frame["PERMNO"] = frame["PERMNO"].astype("int64")
    frame["holding_month"] = frame["holding_month"].astype(str)
    for column in ["MthRet", "MthRetFlg", "MthDelFlg"]:
        frame[column] = frame[column].str.strip()

    signature = (
        frame["MthRet"].fillna("<NA>")
        + "|"
        + frame["MthRetFlg"].fillna("<NA>")
        + "|"
        + frame["MthDelFlg"].fillna("<NA>")
    )
    frame["_signature"] = signature
    distinct = frame.groupby(["PERMNO", "holding_month"])["_signature"].nunique()
    conflict_keys = distinct[distinct.gt(1)].index
    rows_in_window = int(len(frame))
    duplicate_rows = int(frame.duplicated(["PERMNO", "holding_month"]).sum())
    frame = frame.drop_duplicates(["PERMNO", "holding_month"], keep="first").copy()
    conflict_mask = pd.MultiIndex.from_frame(frame[["PERMNO", "holding_month"]]).isin(
        conflict_keys
    )
    frame.loc[conflict_mask, ["MthRet", "MthRetFlg", "MthDelFlg"]] = pd.NA

    frame["crsp_mthret"] = pd.to_numeric(frame["MthRet"], errors="coerce")
    frame = frame.rename(
        columns={"MthRetFlg": "crsp_mthretflg", "MthDelFlg": "crsp_mthdelflg"}
    )[["PERMNO", "holding_month", "crsp_mthret", "crsp_mthretflg", "crsp_mthdelflg"]]
    audit = {
        "crsp_rows_in_holding_window": rows_in_window,
        "crsp_duplicate_rows_collapsed": duplicate_rows,
        "crsp_conflicting_duplicate_keys_set_missing": int(conflict_mask.sum()),
        "crsp_keys": int(len(frame)),
    }
    return frame, audit


def combine(outcomes: pd.DataFrame, crsp: pd.DataFrame) -> pd.DataFrame:
    frame = outcomes.merge(
        crsp, on=["PERMNO", "holding_month"], how="left", validate="one_to_one"
    )
    frame["crsp_delisting_month"] = frame["crsp_mthdelflg"].notna() & frame[
        "crsp_mthdelflg"
    ].ne(NO_DELISTING_FLAG)
    has_crsp = frame["crsp_mthret"].notna()
    has_prof = frame["ret_adjprc"].notna()

    use_delisting = frame["crsp_delisting_month"] & has_crsp
    use_prof = ~use_delisting & has_prof
    use_fill = ~use_delisting & ~has_prof & has_crsp

    frame["ret_final"] = np.select(
        [use_delisting, use_prof, use_fill],
        [frame["crsp_mthret"], frame["ret_adjprc"], frame["crsp_mthret"]],
        default=np.nan,
    )
    frame["ret_source"] = np.select(
        [use_delisting, use_prof, use_fill],
        ["crsp_delisting", "adj_prc", "crsp_fill"],
        default="missing",
    )
    columns = [
        "PERMNO",
        "formation_month",
        "holding_month",
        "EXCHCD",
        "ret_adjprc",
        "crsp_mthret",
        "crsp_mthretflg",
        "crsp_mthdelflg",
        "crsp_delisting_month",
        "ret_final",
        "ret_source",
    ]
    return frame[columns].sort_values(["formation_month", "PERMNO"]).reset_index(drop=True)


def qa_summary(frame: pd.DataFrame) -> dict[str, Any]:
    prof_missing = frame["ret_adjprc"].isna()
    both = frame["ret_adjprc"].notna() & frame["crsp_mthret"].notna()
    normal = both & ~frame["crsp_delisting_month"]
    delist_both = both & frame["crsp_delisting_month"]
    diff = (frame.loc[normal, "crsp_mthret"] - frame.loc[normal, "ret_adjprc"]).abs()

    delisting_by_flag = []
    for flag, group in frame.loc[frame["crsp_delisting_month"]].groupby(
        "crsp_mthdelflg", observed=True
    ):
        paired = group.dropna(subset=["ret_adjprc", "crsp_mthret"])
        delisting_by_flag.append(
            {
                "crsp_mthdelflg": str(flag),
                "rows": int(len(group)),
                "rows_with_crsp_return": int(group["crsp_mthret"].notna().sum()),
                "rows_with_professor_return": int(group["ret_adjprc"].notna().sum()),
                "mean_crsp_return": float(group["crsp_mthret"].mean()),
                "paired_rows": int(len(paired)),
                "paired_mean_professor_return": float(paired["ret_adjprc"].mean())
                if len(paired)
                else None,
                "paired_mean_crsp_return": float(paired["crsp_mthret"].mean())
                if len(paired)
                else None,
            }
        )
    flag_crosstab = (
        frame.loc[frame["crsp_delisting_month"]]
        .groupby(["crsp_mthdelflg", "crsp_mthretflg"], dropna=False, observed=True)
        .size()
    )
    return {
        "rows": int(len(frame)),
        "formation_months": int(frame["formation_month"].nunique()),
        "ret_source_counts": {
            str(key): int(value) for key, value in frame["ret_source"].value_counts().items()
        },
        "professor_return_missing": int(prof_missing.sum()),
        "professor_missing_filled_by_source": {
            str(key): int(value)
            for key, value in frame.loc[prof_missing, "ret_source"].value_counts().items()
        },
        "crsp_delisting_months": int(frame["crsp_delisting_month"].sum()),
        "delisting_month_flag_crosstab": {
            f"{delflg}|{retflg}": int(count)
            for (delflg, retflg), count in flag_crosstab.items()
        },
        "delisting_months_by_flag": delisting_by_flag,
        "non_delisting_months_with_both_returns": {
            "rows": int(normal.sum()),
            "mean_crsp_minus_professor": float(
                (frame.loc[normal, "crsp_mthret"] - frame.loc[normal, "ret_adjprc"]).mean()
            ),
            "median_abs_difference": float(diff.median()),
            "share_abs_difference_below_0.001": float(diff.lt(0.001).mean()),
            "share_abs_difference_below_0.01": float(diff.lt(0.01).mean()),
            "sign_agreement_share": float(
                (
                    np.sign(frame.loc[normal, "crsp_mthret"])
                    == np.sign(frame.loc[normal, "ret_adjprc"])
                ).mean()
            ),
        },
        "delisting_months_with_both_returns": {
            "rows": int(delist_both.sum()),
            "mean_professor_return": float(frame.loc[delist_both, "ret_adjprc"].mean()),
            "mean_crsp_return": float(frame.loc[delist_both, "crsp_mthret"].mean()),
        },
    }


def main() -> None:
    args = parser().parse_args()
    start = str(pd.Period(args.formation_start, freq="M"))
    end = str(pd.Period(args.formation_end, freq="M"))
    professor = load_professor(args.professor_csv)
    outcomes = professor_outcomes(professor, start, end)
    crsp, crsp_audit = load_crsp_monthly(
        args.crsp_monthly,
        str(pd.Period(start, freq="M") + 1),
        str(pd.Period(end, freq="M") + 1),
    )
    supplement = combine(outcomes, crsp)
    if supplement.duplicated(["PERMNO", "formation_month"]).any():
        raise AssertionError("Supplement lost its one-row-per-stock-month grain.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    supplement.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, args.output)

    qa = {
        "formation_period": [start, end],
        "holding_period": [str(pd.Period(start, freq="M") + 1), str(pd.Period(end, freq="M") + 1)],
        "universe": "Professor CSV rows with EXCHCD in {1,2,3} at formation month t",
        "rule": [
            "CRSP MthDelFlg != N and CRSP MthRet present -> CRSP MthRet (crsp_delisting)",
            "else professor exact adjacent-month ADJ_PRC return present -> ADJ_PRC return (adj_prc)",
            "else CRSP MthRet present -> CRSP MthRet (crsp_fill)",
            "else missing; never fill with zero (missing)",
        ],
        "inputs": {
            "professor_csv": args.professor_csv.name,
            "professor_csv_sha256": sha256(args.professor_csv),
            "crsp_monthly": args.crsp_monthly.name,
            "crsp_monthly_sha256": sha256(args.crsp_monthly),
        },
        "output": args.output.name,
        "output_sha256": sha256(args.output),
        **crsp_audit,
        **qa_summary(supplement),
    }
    args.qa.parent.mkdir(parents=True, exist_ok=True)
    args.qa.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
