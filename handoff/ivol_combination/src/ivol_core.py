from __future__ import annotations

import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DAILY_FILES = (
    "CRSP_Daily_1999-12_2004.csv.gz",
    "CRSP_Daily_2005_2009.csv.gz",
    "CRSP_Daily_2010_2014.csv.gz.gz",
    "CRSP_Daily_2015_2020.csv.gz.gz",
)
FF3_FILE = "FF3_Daily.zip"
OFFICIAL_EXCHANGE_CODES = {1, 2, 3}
PROFESSOR_REQUIRED_COLUMNS = ("PERMNO", "date", "EXCHCD")
PROFESSOR_OPTIONAL_COLUMNS = ("TICKER", "COMNAM")


def resolve_data_dir(value: str | Path | None) -> Path:
    candidate = value or os.environ.get("IVOL_DATA_DIR")
    if not candidate:
        raise ValueError("Pass --data-dir or set IVOL_DATA_DIR.")
    path = Path(candidate).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    return path


def ensure_parent(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    output = ensure_parent(path)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def find_daily_files(data_dir: Path) -> list[Path]:
    paths = [data_dir / filename for filename in DAILY_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CRSP daily files: {missing}")
    return paths


def load_ff3(zip_path: str | Path) -> pd.DataFrame:
    zip_path = Path(zip_path).expanduser().resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"Expected one file in {zip_path.name}; found {members}")
        text = archive.read(members[0]).decode("utf-8-sig", errors="replace")

    rows: list[list[str]] = []
    for raw_line in text.splitlines():
        fields = [item.strip() for item in raw_line.split(",")]
        if len(fields) < 5 or len(fields[0]) != 8 or not fields[0].isdigit():
            continue
        rows.append(fields[:5])
    if not rows:
        raise ValueError(f"No daily FF3 rows found in {zip_path}")

    frame = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="raise")
    for column in ["mkt_rf", "smb", "hml", "rf"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise") / 100.0
    if frame["date"].duplicated().any():
        raise ValueError("FF3 contains duplicate daily dates.")
    return frame.sort_values("date").reset_index(drop=True)


def load_professor_stock_data(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Professor stock file does not exist: {source}")

    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = sorted(set(PROFESSOR_REQUIRED_COLUMNS).difference(header))
    if missing:
        raise ValueError(f"Professor stock file is missing required columns: {missing}")
    usecols = [
        *PROFESSOR_REQUIRED_COLUMNS,
        *(column for column in PROFESSOR_OPTIONAL_COLUMNS if column in header),
    ]
    frame = pd.read_csv(
        source,
        usecols=usecols,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    raw_rows = len(frame)
    for column in PROFESSOR_OPTIONAL_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame["PERMNO"] = pd.to_numeric(frame["PERMNO"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["EXCHCD"] = pd.to_numeric(frame["EXCHCD"], errors="coerce")
    invalid_keys = frame["PERMNO"].isna() | frame["date"].isna()
    if invalid_keys.any():
        raise ValueError(
            "Professor stock file has "
            f"{int(invalid_keys.sum()):,} invalid PERMNO/date rows."
        )
    frame["PERMNO"] = frame["PERMNO"].astype("int64")
    frame["EXCHCD"] = frame["EXCHCD"].astype("Int64")
    frame["formation_month"] = frame["date"].dt.to_period("M").astype(str)
    duplicate_keys = int(frame.duplicated(["PERMNO", "formation_month"]).sum())
    if duplicate_keys:
        raise ValueError(
            "Professor stock file must have one row per PERMNO/month; found "
            f"{duplicate_keys:,} duplicate keys."
        )
    frame = frame.sort_values(["formation_month", "PERMNO"]).reset_index(drop=True)
    audit = {
        "source_file": source.name,
        "raw_rows": int(raw_rows),
        "unique_permnos": int(frame["PERMNO"].nunique()),
        "month_start": str(frame["formation_month"].min()),
        "month_end": str(frame["formation_month"].max()),
        "duplicate_permno_month_keys": duplicate_keys,
        "exchange_code_counts": {
            str(key): int(value)
            for key, value in frame["EXCHCD"].value_counts(dropna=False).items()
        },
    }
    return frame, audit


def official_formation_universe(
    professor: pd.DataFrame,
    formation_start: str | pd.Period,
    formation_end: str | pd.Period,
) -> pd.DataFrame:
    start = str(pd.Period(formation_start, freq="M"))
    end = str(pd.Period(formation_end, freq="M"))
    mask = professor["formation_month"].between(start, end) & professor[
        "EXCHCD"
    ].isin(OFFICIAL_EXCHANGE_CODES)
    result = professor.loc[mask].copy()
    if result.empty:
        raise ValueError("The professor formation universe is empty.")
    if result.duplicated(["PERMNO", "formation_month"]).any():
        raise ValueError("Formation universe has duplicate PERMNO/month keys.")
    return result


def ivol_from_moments(
    xtx: np.ndarray,
    xty: np.ndarray,
    yty: float,
    n_obs: int,
) -> tuple[float, int]:
    """Return the OLS residual sample standard deviation and matrix rank."""

    rank = int(np.linalg.matrix_rank(xtx))
    if rank < xtx.shape[0] or n_obs <= 1:
        return math.nan, rank
    beta = np.linalg.solve(xtx, xty)
    sse = float(yty - beta @ xty)
    tolerance = 1e-12 * max(1.0, abs(float(yty)))
    if sse < -tolerance:
        raise FloatingPointError(f"Materially negative SSE: {sse}")
    return math.sqrt(max(0.0, sse) / (n_obs - 1)), rank


def assign_deciles(
    frame: pd.DataFrame,
    month_column: str = "formation_month",
    signal_column: str = "ivol_daily",
    id_column: str = "PERMNO",
) -> pd.Series:
    """Deterministic equal-count deciles; ties break by identifier ascending."""

    result = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for _, index in frame.groupby(month_column, sort=True).groups.items():
        ordered = frame.loc[index].sort_values(
            [signal_column, id_column], kind="mergesort"
        )
        n_rows = len(ordered)
        bins = (np.arange(n_rows, dtype=np.int64) * 10 // n_rows) + 1
        result.loc[ordered.index] = bins
    return result
