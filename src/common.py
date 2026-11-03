from __future__ import annotations

import json
import math
import os
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DAILY_FILES = (
    "CRSP_Daily_1999-12_2004.csv.gz",
    "CRSP_Daily_2005_2009.csv.gz",
    "CRSP_Daily_2010_2014.csv.gz.gz",
    "CRSP_Daily_2015_2020.csv.gz.gz",
)
MONTHLY_FILE = "CRSP_Monthly_1999-12_2020.csv.gz"
FF3_FILE = "FF3_Daily.zip"

UNIVERSE_RULES = {
    "ConditionalType": {"RW"},
    "USIncFlg": {"Y"},
    "IssuerType": {"CORP"},
    "SecurityType": {"EQTY"},
    "SecuritySubType": {"COM"},
    "ShareType": {"NS"},
    "PrimaryExch": {"N", "A", "Q"},
}


def resolve_data_dir(value: str | Path | None) -> Path:
    candidate = value or os.environ.get("IVOL_DATA_DIR")
    if not candidate:
        raise ValueError("Pass --data-dir or set IVOL_DATA_DIR.")
    path = Path(candidate).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    return path


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def write_json(payload: dict, path: str | Path) -> None:
    output = ensure_parent(path)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def apply_universe_filter(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    mask = pd.Series(True, index=frame.index)
    audit: dict[str, int] = {"input_rows": int(len(frame))}
    for column, accepted in UNIVERSE_RULES.items():
        mask &= frame[column].astype("string").str.strip().isin(accepted)
        audit[f"rows_after_{column}"] = int(mask.sum())
    return frame.loc[mask].copy(), audit


def load_ff3(zip_path: str | Path) -> pd.DataFrame:
    zip_path = Path(zip_path)
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


def next_month_label(month: str | pd.Period) -> str:
    return str(pd.Period(month, freq="M") + 1)


def assign_deciles(
    frame: pd.DataFrame,
    month_column: str = "formation_month",
    signal_column: str = "ivol_daily",
    id_column: str = "PERMNO",
) -> pd.Series:
    """Deterministic equal-count deciles; ties break by PERMNO ascending."""
    result = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for _, index in frame.groupby(month_column, sort=True).groups.items():
        ordered = frame.loc[index].sort_values(
            [signal_column, id_column], kind="mergesort"
        )
        n_rows = len(ordered)
        if n_rows == 0:
            continue
        bins = (np.arange(n_rows, dtype=np.int64) * 10 // n_rows) + 1
        result.loc[ordered.index] = bins
    return result


def ivol_from_moments(
    xtx: np.ndarray,
    xty: np.ndarray,
    yty: float,
    n_obs: int,
) -> tuple[float, int]:
    """Return residual sample SD (ddof=1) and design-matrix rank."""
    rank = int(np.linalg.matrix_rank(xtx))
    if rank < xtx.shape[0] or n_obs <= 1:
        return math.nan, rank
    beta = np.linalg.solve(xtx, xty)
    sse = float(yty - beta @ xty)
    tolerance = 1e-12 * max(1.0, abs(float(yty)))
    if sse < -tolerance:
        raise FloatingPointError(f"Materially negative SSE: {sse}")
    sse = max(0.0, sse)
    return math.sqrt(sse / (n_obs - 1)), rank


def find_daily_files(data_dir: Path) -> list[Path]:
    paths = [data_dir / filename for filename in DAILY_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CRSP daily files: {missing}")
    return paths


def compact_counts(series: pd.Series, limit: int | None = None) -> dict[str, int]:
    counts = series.fillna("<MISSING>").astype(str).value_counts(dropna=False)
    if limit is not None:
        counts = counts.head(limit)
    return {str(key): int(value) for key, value in counts.items()}
