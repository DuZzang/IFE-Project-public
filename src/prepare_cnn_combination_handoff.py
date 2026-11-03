from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = [
    "PERMNO",
    "formation_date",
    "formation_month",
    "holding_month",
    "cnn_prob_up",
    "cnn_rank_pct",
    "cnn_score_signed",
    "cnn_decile",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a leakage-safe CNN signal file for the combination strategy."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.source, usecols=SIGNAL_COLUMNS)
    frame["PERMNO"] = pd.to_numeric(frame["PERMNO"], errors="raise").astype("int64")
    frame["formation_date"] = pd.to_datetime(
        frame["formation_date"], errors="raise"
    )
    frame["formation_month"] = frame["formation_month"].astype(str)
    frame["holding_month"] = frame["holding_month"].astype(str)
    frame["cnn_decile"] = pd.to_numeric(
        frame["cnn_decile"], errors="raise"
    ).astype("int8")
    for column in ("cnn_prob_up", "cnn_rank_pct", "cnn_score_signed"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")

    duplicate_keys = int(frame.duplicated(["PERMNO", "formation_month"]).sum())
    if duplicate_keys:
        raise ValueError(f"Duplicate PERMNO-formation_month keys: {duplicate_keys}")
    if not frame["cnn_prob_up"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("cnn_prob_up is outside [0, 1]")
    if not frame["cnn_rank_pct"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("cnn_rank_pct is outside [0, 1]")
    if not frame["cnn_score_signed"].between(-1.0, 1.0, inclusive="both").all():
        raise ValueError("cnn_score_signed is outside [-1, 1]")
    if not frame["cnn_decile"].between(1, 10, inclusive="both").all():
        raise ValueError("cnn_decile is outside 1..10")

    formation_period = pd.PeriodIndex(frame["formation_month"], freq="M")
    holding_period = pd.PeriodIndex(frame["holding_month"], freq="M")
    timing_mismatch = int((holding_period != formation_period + 1).sum())
    if timing_mismatch:
        raise ValueError(f"holding_month is not t+1 for {timing_mismatch} rows")
    date_month_mismatch = int(
        (
            frame["formation_date"].dt.to_period("M").astype(str)
            != frame["formation_month"]
        ).sum()
    )
    if date_month_mismatch:
        raise ValueError(
            f"formation_date and formation_month differ for {date_month_mismatch} rows"
        )

    expected_signed = 2.0 * frame["cnn_rank_pct"] - 1.0
    signed_error = float(
        np.max(np.abs(frame["cnn_score_signed"].to_numpy() - expected_signed.to_numpy()))
    )
    if signed_error > 1e-12:
        raise ValueError(f"cnn_score_signed identity failed: max error {signed_error}")

    frame = frame.sort_values(
        ["formation_month", "PERMNO"], kind="mergesort"
    ).reset_index(drop=True)
    frame["formation_date"] = frame["formation_date"].dt.strftime("%Y-%m-%d")
    frame.to_csv(
        args.output,
        index=False,
        encoding="utf-8",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        float_format="%.10g",
    )

    monthly_counts = frame.groupby("formation_month", sort=True).size()
    summary = {
        "status": "validated",
        "source_filename": args.source.name,
        "output_filename": args.output.name,
        "rows": int(len(frame)),
        "unique_permnos": int(frame["PERMNO"].nunique()),
        "formation_month_start": str(frame["formation_month"].min()),
        "formation_month_end": str(frame["formation_month"].max()),
        "holding_month_start": str(frame["holding_month"].min()),
        "holding_month_end": str(frame["holding_month"].max()),
        "formation_month_count": int(frame["formation_month"].nunique()),
        "duplicate_permno_month_keys": duplicate_keys,
        "holding_month_t_plus_1_mismatches": timing_mismatch,
        "formation_date_month_mismatches": date_month_mismatch,
        "score_ranges": {
            "cnn_prob_up": [
                float(frame["cnn_prob_up"].min()),
                float(frame["cnn_prob_up"].max()),
            ],
            "cnn_rank_pct": [
                float(frame["cnn_rank_pct"].min()),
                float(frame["cnn_rank_pct"].max()),
            ],
            "cnn_score_signed": [
                float(frame["cnn_score_signed"].min()),
                float(frame["cnn_score_signed"].max()),
            ],
            "cnn_decile": [
                int(frame["cnn_decile"].min()),
                int(frame["cnn_decile"].max()),
            ],
        },
        "cnn_score_signed_identity_max_abs_error": signed_error,
        "monthly_row_count": {
            "min": int(monthly_counts.min()),
            "mean": float(monthly_counts.mean()),
            "max": int(monthly_counts.max()),
        },
        "output_sha256": sha256_file(args.output),
        "recommended_join_key": ["PERMNO", "formation_month"],
        "recommended_signal": "cnn_score_signed",
        "timing_rule": "Use the formation-month t signal to form the portfolio for holding month t+1.",
        "contains_realized_returns": False,
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
