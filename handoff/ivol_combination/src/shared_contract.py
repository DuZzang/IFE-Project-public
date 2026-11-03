from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OFFICIAL_EXCHANGE_CODES = {1, 2, 3}
PROFESSOR_REQUIRED_COLUMNS = ("PERMNO", "date", "EXCHCD", "ADJ_PRC")
PROFESSOR_OPTIONAL_COLUMNS = ("TICKER", "COMNAM")


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


def canonical_month(values: pd.Series, label: str) -> pd.Series:
    """Return YYYY-MM for full dates and common CSV month encodings."""

    normalized = values.astype("string").str.strip().str.replace(
        r"^(\d{4})(\d{2})(?:\.0+)?$", r"\1-\2", regex=True
    )
    parsed = pd.to_datetime(normalized, errors="coerce", format="mixed")
    if parsed.isna().any():
        examples = values.loc[parsed.isna()].astype(str).head(5).tolist()
        raise ValueError(f"{label} contains invalid month values, examples: {examples}")
    return parsed.dt.to_period("M").astype(str)


def load_professor_prices(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Professor CSV does not exist: {source}")
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = sorted(set(PROFESSOR_REQUIRED_COLUMNS).difference(header))
    if missing:
        raise ValueError(f"Professor CSV is missing required columns: {missing}")
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
    for column in PROFESSOR_OPTIONAL_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame["PERMNO"] = pd.to_numeric(frame["PERMNO"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["EXCHCD"] = pd.to_numeric(frame["EXCHCD"], errors="coerce")
    frame["ADJ_PRC"] = pd.to_numeric(frame["ADJ_PRC"], errors="coerce")
    invalid_key = frame["PERMNO"].isna() | frame["date"].isna()
    if invalid_key.any():
        raise ValueError(f"Professor CSV has {int(invalid_key.sum()):,} invalid key rows")
    frame["PERMNO"] = frame["PERMNO"].astype("int64")
    frame["EXCHCD"] = frame["EXCHCD"].astype("Int64")
    frame["formation_month"] = frame["date"].dt.to_period("M").astype(str)
    duplicate_keys = int(frame.duplicated(["PERMNO", "formation_month"]).sum())
    if duplicate_keys:
        raise ValueError(
            "Professor CSV must be one row per PERMNO/month; found "
            f"{duplicate_keys:,} duplicate keys"
        )
    frame = frame.sort_values(["formation_month", "PERMNO"]).reset_index(drop=True)
    audit = {
        "source_file": source.name,
        "rows": int(len(frame)),
        "unique_permnos": int(frame["PERMNO"].nunique()),
        "month_start": str(frame["formation_month"].min()),
        "month_end": str(frame["formation_month"].max()),
        "duplicate_keys": duplicate_keys,
        "missing_adj_prc": int(frame["ADJ_PRC"].isna().sum()),
        "nonpositive_adj_prc": int(frame["ADJ_PRC"].le(0).fillna(False).sum()),
    }
    return frame, audit


def official_formation_universe(
    professor: pd.DataFrame,
    formation_start: str,
    formation_end: str,
) -> pd.DataFrame:
    start = str(pd.Period(formation_start, freq="M"))
    end = str(pd.Period(formation_end, freq="M"))
    if pd.Period(end, freq="M") < pd.Period(start, freq="M"):
        raise ValueError("formation_end cannot precede formation_start")
    mask = professor["formation_month"].between(start, end) & professor[
        "EXCHCD"
    ].isin(OFFICIAL_EXCHANGE_CODES)
    result = professor.loc[mask].copy()
    if result.empty:
        raise ValueError("Professor formation universe is empty")
    if result.duplicated(["PERMNO", "formation_month"]).any():
        raise AssertionError("Formation universe has duplicate keys")
    return result


def build_adj_price_outcomes(
    professor: pd.DataFrame,
    formation_start: str,
    formation_end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build exact t-to-t+1 price returns without an outcome-month survivor filter."""

    formation = official_formation_universe(
        professor, formation_start=formation_start, formation_end=formation_end
    ).rename(
        columns={
            "date": "formation_date",
            "ADJ_PRC": "formation_adj_prc",
        }
    )
    formation["holding_month"] = (
        pd.PeriodIndex(formation["formation_month"], freq="M") + 1
    ).astype(str)
    future = professor[
        ["PERMNO", "formation_month", "date", "ADJ_PRC", "EXCHCD"]
    ].rename(
        columns={
            "formation_month": "holding_month",
            "date": "holding_date",
            "ADJ_PRC": "holding_adj_prc",
            "EXCHCD": "holding_exchcd",
        }
    )
    outcomes = formation.merge(
        future,
        on=["PERMNO", "holding_month"],
        how="left",
        validate="one_to_one",
    )
    valid_prices = outcomes["formation_adj_prc"].gt(0) & outcomes[
        "holding_adj_prc"
    ].gt(0)
    outcomes["next_month_return"] = np.where(
        valid_prices,
        outcomes["holding_adj_prc"] / outcomes["formation_adj_prc"] - 1.0,
        np.nan,
    )
    expected_holding = (
        pd.PeriodIndex(outcomes["formation_month"], freq="M") + 1
    ).astype(str)
    if not outcomes["holding_month"].eq(expected_holding).all():
        raise AssertionError("holding_month is not formation_month + 1")
    columns = [
        "PERMNO",
        "formation_date",
        "formation_month",
        "holding_month",
        "EXCHCD",
        "TICKER",
        "COMNAM",
        "formation_adj_prc",
        "holding_date",
        "holding_exchcd",
        "holding_adj_prc",
        "next_month_return",
    ]
    audit = {
        "formation_rows": int(len(outcomes)),
        "formation_months": int(outcomes["formation_month"].nunique()),
        "holding_month_start": str(outcomes["holding_month"].min()),
        "holding_month_end": str(outcomes["holding_month"].max()),
        "rows_with_next_month_return": int(outcomes["next_month_return"].notna().sum()),
        "rows_missing_next_month_return": int(outcomes["next_month_return"].isna().sum()),
        "return_definition": "ADJ_PRC(t+1) / ADJ_PRC(t) - 1 using exact adjacent month",
        "outcome_month_survivor_refilter": False,
        "missing_policy": "leave missing; never fill zero",
    }
    return outcomes[columns], audit


def load_formation_caps(
    monthly_path: str | Path,
    formation_keys: pd.DataFrame,
    chunksize: int = 500_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load CRSP formation-month capitalization for optional value weighting."""

    keys = formation_keys[["PERMNO", "formation_month"]].drop_duplicates().copy()
    if len(keys) != len(formation_keys):
        raise ValueError("Formation keys supplied for cap loading are not unique")
    keys["PERMNO"] = pd.to_numeric(keys["PERMNO"], errors="raise").astype("int64")
    keys["formation_month"] = keys["formation_month"].astype(str)
    start = keys["formation_month"].min()
    end = keys["formation_month"].max()
    pieces: list[pd.DataFrame] = []
    source_rows_in_window = 0
    for chunk in pd.read_csv(
        monthly_path,
        compression="infer",
        usecols=["PERMNO", "MthCalDt", "MthCap"],
        chunksize=chunksize,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    ):
        dates = pd.to_datetime(chunk["MthCalDt"], errors="coerce")
        months = dates.dt.to_period("M").astype("string")
        in_window = months.between(start, end)
        if not in_window.any():
            continue
        selected = chunk.loc[in_window, ["PERMNO", "MthCap"]].copy()
        selected["formation_month"] = months.loc[in_window].astype(str).to_numpy()
        selected["PERMNO"] = pd.to_numeric(selected["PERMNO"], errors="coerce")
        selected["MthCap"] = pd.to_numeric(selected["MthCap"], errors="coerce")
        selected = selected.dropna(subset=["PERMNO"]).copy()
        selected["PERMNO"] = selected["PERMNO"].astype("int64")
        source_rows_in_window += int(len(selected))
        selected = selected.merge(
            keys,
            on=["PERMNO", "formation_month"],
            how="inner",
            validate="many_to_one",
        )
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise RuntimeError("No CRSP market-cap rows matched the formation keys")
    caps = pd.concat(pieces, ignore_index=True)
    duplicate_mask = caps.duplicated(["PERMNO", "formation_month"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = caps.loc[duplicate_mask].copy()
        duplicate_rows["_signature"] = duplicate_rows["MthCap"].astype("string").fillna(
            "<MISSING>"
        )
        conflicts = duplicate_rows.groupby(
            ["PERMNO", "formation_month"], observed=True
        )["_signature"].nunique(dropna=False)
        if conflicts.gt(1).any():
            raise ValueError("CRSP monthly data has conflicting MthCap duplicate rows")
    duplicates_removed = int(
        caps.duplicated(["PERMNO", "formation_month"], keep="first").sum()
    )
    caps = caps.drop_duplicates(
        ["PERMNO", "formation_month"], keep="first"
    ).rename(columns={"MthCap": "formation_market_cap"})
    caps = caps[["PERMNO", "formation_month", "formation_market_cap"]]
    audit = {
        "monthly_source_rows_in_window": source_rows_in_window,
        "matched_cap_keys": int(len(caps)),
        "formation_keys_without_cap_row": int(len(keys) - len(caps)),
        "duplicate_cap_rows_collapsed": duplicates_removed,
        "missing_cap_values": int(caps["formation_market_cap"].isna().sum()),
        "nonpositive_cap_values": int(
            caps["formation_market_cap"].le(0).fillna(False).sum()
        ),
    }
    return caps, audit

