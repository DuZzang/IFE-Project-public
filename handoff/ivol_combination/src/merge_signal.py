from __future__ import annotations

import argparse

import pandas as pd

from ivol_core import ensure_parent, write_json


SIGNAL_VALUE_COLUMNS = [
    "ivol_daily",
    "ivol_annualized",
    "n_daily_obs",
    "regression_rank",
    "ivol_decile",
    "ivol_rank_pct",
    "low_ivol_score",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Merge the IVOL signal into a one-row-per-stock-month factor panel."
    )
    result.add_argument("--panel", required=True)
    result.add_argument("--signal", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--log", required=True)
    result.add_argument("--id-column", default="PERMNO")
    result.add_argument("--panel-month-column", default="formation_month")
    result.add_argument(
        "--signal-month-column",
        choices=["formation_month", "holding_month"],
        default="formation_month",
        help=(
            "Use formation_month when the panel is dated at signal formation t; "
            "use holding_month when panel rows represent the investable month t+1."
        ),
    )
    return result


def canonical_month(values: pd.Series, label: str) -> pd.Series:
    # CSV exports often encode a month as 202001 (integer), 202001.0 (Excel),
    # 2020-01, or a full date. Convert the compact YYYYMM variants explicitly;
    # pandas otherwise interprets numeric 202001 as nanoseconds after 1970-01-01.
    normalized = values.astype("string").str.strip().str.replace(
        r"^(\d{4})(\d{2})(?:\.0+)?$", r"\1-\2", regex=True
    )
    parsed = pd.to_datetime(normalized, errors="coerce", format="mixed")
    if parsed.isna().any():
        examples = values.loc[parsed.isna()].astype(str).head(5).tolist()
        raise ValueError(f"{label} has invalid month values, examples: {examples}")
    return parsed.dt.to_period("M").astype(str)


def merge_signal(
    panel: pd.DataFrame,
    signal: pd.DataFrame,
    *,
    id_column: str,
    panel_month_column: str,
    signal_month_column: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    for column in [id_column, panel_month_column]:
        if column not in panel:
            raise ValueError(f"Factor panel is missing column: {column}")
    signal_required = {"PERMNO", signal_month_column, *SIGNAL_VALUE_COLUMNS}
    missing_signal = sorted(signal_required.difference(signal.columns))
    if missing_signal:
        raise ValueError(f"IVOL signal is missing columns: {missing_signal}")
    collisions = sorted(set(SIGNAL_VALUE_COLUMNS).intersection(panel.columns))
    if collisions:
        raise ValueError(
            "Factor panel already contains IVOL output columns; remove or rename them "
            f"before merging: {collisions}"
        )

    panel = panel.copy()
    panel[id_column] = pd.to_numeric(panel[id_column], errors="raise").astype("int64")
    panel["_ivol_join_month"] = canonical_month(
        panel[panel_month_column], panel_month_column
    )
    if panel.duplicated([id_column, "_ivol_join_month"]).any():
        raise ValueError("Factor panel is not one row per stock/month.")

    selected = signal[["PERMNO", signal_month_column, *SIGNAL_VALUE_COLUMNS]].copy()
    selected["PERMNO"] = pd.to_numeric(selected["PERMNO"], errors="raise").astype(
        "int64"
    )
    selected["_ivol_join_month"] = canonical_month(
        selected[signal_month_column], signal_month_column
    )
    if selected.duplicated(["PERMNO", "_ivol_join_month"]).any():
        raise ValueError("IVOL signal is not unique on the requested join month.")
    selected = selected.drop(columns=[signal_month_column])
    if id_column != "PERMNO":
        selected = selected.rename(columns={"PERMNO": id_column})

    merged = panel.merge(
        selected,
        on=[id_column, "_ivol_join_month"],
        how="left",
        validate="one_to_one",
    )
    matched = int(merged["ivol_daily"].notna().sum())
    audit: dict[str, float | int | str] = {
        "panel_rows": int(len(panel)),
        "matched_ivol_rows": matched,
        "unmatched_panel_rows": int(len(panel) - matched),
        "match_rate": float(matched / len(panel)) if len(panel) else 0.0,
        "panel_month_column": panel_month_column,
        "signal_month_column": signal_month_column,
        "missing_policy": "Unmatched IVOL remains missing; never fill with zero",
    }
    return merged.drop(columns=["_ivol_join_month"]), audit


def main() -> None:
    args = parser().parse_args()
    panel = pd.read_csv(args.panel, compression="infer", low_memory=False)
    signal = pd.read_csv(args.signal, compression="infer", low_memory=False)
    merged, audit = merge_signal(
        panel,
        signal,
        id_column=args.id_column,
        panel_month_column=args.panel_month_column,
        signal_month_column=args.signal_month_column,
    )
    output = ensure_parent(args.output)
    merged.to_csv(output, index=False, compression="infer")
    write_json(audit, args.log)
    print(
        f"Wrote {len(merged):,} rows to {output}; "
        f"IVOL match rate={audit['match_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
