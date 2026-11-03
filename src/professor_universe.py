from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import assign_deciles


OFFICIAL_EXCHANGE_CODES = {1, 2, 3}
PROFESSOR_COLUMNS = [
    "PERMNO",
    "date",
    "EXCHCD",
    "TICKER",
    "COMNAM",
    "ADJ_PRC",
]


def load_professor_stock_data(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load and validate the professor-supplied stock-month file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Professor stock file does not exist: {source}")

    frame = pd.read_csv(
        source,
        usecols=PROFESSOR_COLUMNS,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    raw_rows = len(frame)
    frame["PERMNO"] = pd.to_numeric(frame["PERMNO"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["EXCHCD"] = pd.to_numeric(frame["EXCHCD"], errors="coerce")
    frame["ADJ_PRC"] = pd.to_numeric(frame["ADJ_PRC"], errors="coerce")

    invalid_keys = frame["PERMNO"].isna() | frame["date"].isna()
    if invalid_keys.any():
        raise ValueError(
            "Professor stock file has "
            f"{int(invalid_keys.sum()):,} rows with an invalid PERMNO or date."
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
        "missing_adj_prc": int(frame["ADJ_PRC"].isna().sum()),
        "nonpositive_adj_prc": int(frame["ADJ_PRC"].le(0).fillna(False).sum()),
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
    """Return the authoritative month-t universe (EXCHCD 1/2/3 only)."""
    start = str(pd.Period(formation_start, freq="M"))
    end = str(pd.Period(formation_end, freq="M"))
    mask = professor["formation_month"].between(start, end) & professor[
        "EXCHCD"
    ].isin(OFFICIAL_EXCHANGE_CODES)
    result = professor.loc[mask].copy()
    if result.empty:
        raise ValueError("The official professor formation universe is empty.")
    if result.duplicated(["PERMNO", "formation_month"]).any():
        raise ValueError("Official formation universe has duplicate PERMNO/month keys.")
    return result


def load_formation_caps(
    monthly_path: str | Path,
    formation_keys: pd.DataFrame,
    chunksize: int,
) -> tuple[pd.DataFrame, dict]:
    """Load CRSP month-t market cap for optional value weighting."""
    keys = formation_keys[["PERMNO", "formation_month"]].drop_duplicates().copy()
    if len(keys) != len(formation_keys):
        raise ValueError("Formation keys supplied for cap loading are not unique.")
    keys["PERMNO"] = pd.to_numeric(keys["PERMNO"], errors="raise").astype("int64")
    keys["formation_month"] = keys["formation_month"].astype(str)
    start = keys["formation_month"].min()
    end = keys["formation_month"].max()

    pieces: list[pd.DataFrame] = []
    source_rows_in_window = 0
    for chunk in pd.read_csv(
        monthly_path,
        compression="gzip",
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
        selected = selected.dropna(subset=["PERMNO"]).copy()
        selected["PERMNO"] = selected["PERMNO"].astype("int64")
        selected["MthCap"] = pd.to_numeric(selected["MthCap"], errors="coerce")
        source_rows_in_window += len(selected)
        selected = selected.merge(
            keys,
            on=["PERMNO", "formation_month"],
            how="inner",
            validate="many_to_one",
        )
        if not selected.empty:
            pieces.append(selected)

    if not pieces:
        raise RuntimeError("No CRSP market-cap rows matched the professor universe.")
    caps = pd.concat(pieces, ignore_index=True)
    duplicate_mask = caps.duplicated(["PERMNO", "formation_month"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = caps.loc[duplicate_mask].copy()
        duplicate_rows["_cap_signature"] = (
            duplicate_rows["MthCap"].astype("string").fillna("<MISSING>")
        )
        conflicts = duplicate_rows.groupby(
            ["PERMNO", "formation_month"], observed=True
        )["_cap_signature"].nunique(dropna=False)
        if conflicts.gt(1).any():
            raise ValueError("CRSP monthly data has conflicting MthCap duplicate rows.")
    duplicates_removed = int(
        caps.duplicated(["PERMNO", "formation_month"], keep="first").sum()
    )
    caps = caps.drop_duplicates(
        ["PERMNO", "formation_month"], keep="first"
    ).rename(columns={"MthCap": "formation_market_cap"})
    caps = caps[["PERMNO", "formation_month", "formation_market_cap"]]
    audit = {
        "monthly_source_rows_in_window": int(source_rows_in_window),
        "matched_cap_keys": int(len(caps)),
        "formation_keys_without_cap_row": int(len(keys) - len(caps)),
        "duplicate_cap_rows_collapsed": duplicates_removed,
        "missing_cap_values": int(caps["formation_market_cap"].isna().sum()),
        "nonpositive_cap_values": int(
            caps["formation_market_cap"].le(0).fillna(False).sum()
        ),
    }
    return caps, audit


def build_professor_assignments(
    professor: pd.DataFrame,
    ivol: pd.DataFrame,
    formation_caps: pd.DataFrame,
    formation_start: str | pd.Period,
    formation_end: str | pd.Period,
) -> tuple[pd.DataFrame, dict]:
    """Attach IVOL, form deciles, then attach exact t+1 adjusted-price returns."""
    formation = official_formation_universe(
        professor, formation_start=formation_start, formation_end=formation_end
    )
    formation = formation.rename(
        columns={
            "date": "formation_date",
            "ADJ_PRC": "formation_adj_prc",
        }
    )

    ivol = ivol.copy()
    ivol["PERMNO"] = pd.to_numeric(ivol["PERMNO"], errors="raise").astype("int64")
    ivol["formation_month"] = ivol["formation_month"].astype(str)
    ivol["ivol_daily"] = pd.to_numeric(ivol["ivol_daily"], errors="coerce")
    if ivol.duplicated(["PERMNO", "formation_month"]).any():
        raise ValueError("IVOL input has duplicate PERMNO/formation_month keys.")

    key_check = ivol[["PERMNO", "formation_month"]].merge(
        formation[["PERMNO", "formation_month"]],
        on=["PERMNO", "formation_month"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    ivol_outside_professor = int(key_check["_merge"].ne("both").sum())
    if ivol_outside_professor:
        raise ValueError(
            f"IVOL input contains {ivol_outside_professor:,} keys outside the professor universe."
        )

    assignments = formation.merge(
        ivol,
        on=["PERMNO", "formation_month"],
        how="left",
        validate="one_to_one",
    )
    missing_ivol = int(assignments["ivol_daily"].isna().sum())
    assignments = assignments.dropna(subset=["ivol_daily"]).copy()
    assignments["decile"] = assign_deciles(assignments)

    assignments["next_month"] = (
        pd.PeriodIndex(assignments["formation_month"], freq="M") + 1
    ).astype(str)
    outcome = professor[
        ["PERMNO", "formation_month", "date", "ADJ_PRC", "EXCHCD"]
    ].rename(
        columns={
            "formation_month": "next_month",
            "date": "next_month_date",
            "ADJ_PRC": "next_month_adj_prc",
            "EXCHCD": "next_month_exchcd",
        }
    )
    assignments = assignments.merge(
        outcome,
        on=["PERMNO", "next_month"],
        how="left",
        validate="one_to_one",
    )
    assignments = assignments.merge(
        formation_caps,
        on=["PERMNO", "formation_month"],
        how="left",
        validate="one_to_one",
    )

    valid_price_pair = (
        assignments["formation_adj_prc"].gt(0)
        & assignments["next_month_adj_prc"].gt(0)
    )
    assignments["next_month_return"] = np.where(
        valid_price_pair,
        assignments["next_month_adj_prc"] / assignments["formation_adj_prc"] - 1.0,
        np.nan,
    )

    expected_next = (
        pd.PeriodIndex(assignments["formation_month"], freq="M") + 1
    ).astype(str)
    if not np.array_equal(assignments["next_month"].to_numpy(), expected_next.to_numpy()):
        raise AssertionError("A next_month label is not exactly formation_month + 1.")
    if assignments.duplicated(["PERMNO", "formation_month"]).any():
        raise AssertionError("Assignments lost their one-row-per-stock-month grain.")

    missing_by_decile = (
        assignments.assign(
            missing_return=assignments["next_month_return"].isna().astype(int)
        )
        .groupby("decile", observed=True)
        .agg(rows=("PERMNO", "size"), missing_returns=("missing_return", "sum"))
        .reset_index()
    )
    missing_by_decile["missing_rate"] = (
        missing_by_decile["missing_returns"] / missing_by_decile["rows"]
    )
    audit = {
        "formation_start": str(pd.Period(formation_start, freq="M")),
        "formation_end": str(pd.Period(formation_end, freq="M")),
        "return_start": str(pd.Period(formation_start, freq="M") + 1),
        "return_end": str(pd.Period(formation_end, freq="M") + 1),
        "official_exchange_codes": sorted(OFFICIAL_EXCHANGE_CODES),
        "professor_formation_rows": int(len(formation)),
        "professor_formation_unique_permnos": int(formation["PERMNO"].nunique()),
        "ivol_rows": int(len(ivol)),
        "ivol_keys_outside_professor_universe": ivol_outside_professor,
        "formation_rows_without_valid_ivol": missing_ivol,
        "assignment_rows": int(len(assignments)),
        "rows_with_exact_next_month_price_return": int(
            assignments["next_month_return"].notna().sum()
        ),
        "rows_missing_exact_next_month_price_return": int(
            assignments["next_month_return"].isna().sum()
        ),
        "rows_with_positive_formation_cap": int(
            assignments["formation_market_cap"].gt(0).sum()
        ),
        "return_definition": "ADJ_PRC(t+1) / ADJ_PRC(t) - 1 using an exact calendar-month self-join",
        "return_missing_policy": "Exclude from that month's portfolio return; never replace with zero",
        "outcome_exchange_policy": "Do not refilter t+1 EXCHCD (avoids survivorship bias)",
        "decile_tie_rule": "IVOL ascending, then PERMNO ascending; floor(rank*10/n)+1",
        "missing_returns_by_decile": missing_by_decile.to_dict(orient="records"),
    }

    columns = [
        "PERMNO",
        "formation_date",
        "formation_month",
        "EXCHCD",
        "TICKER",
        "COMNAM",
        "formation_adj_prc",
        "ivol_daily",
        "ivol_annualized",
        "n_daily_obs",
        "regression_rank",
        "decile",
        "next_month",
        "next_month_date",
        "next_month_exchcd",
        "next_month_adj_prc",
        "next_month_return",
        "formation_market_cap",
    ]
    return (
        assignments[columns]
        .sort_values(["formation_month", "decile", "PERMNO"])
        .reset_index(drop=True),
        audit,
    )
