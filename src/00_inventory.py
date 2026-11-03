from __future__ import annotations

import argparse
import gzip
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    DAILY_FILES,
    FF3_FILE,
    MONTHLY_FILE,
    compact_counts,
    ensure_parent,
    load_ff3,
    resolve_data_dir,
    write_json,
)


DAILY_CATEGORIES = [
    "PrimaryExch",
    "ConditionalType",
    "USIncFlg",
    "IssuerType",
    "SecurityType",
    "SecuritySubType",
    "ShareType",
    "DlyRetMissFlg",
    "DlyDelFlg",
]
MONTHLY_CATEGORIES = [
    "PrimaryExch",
    "ConditionalType",
    "USIncFlg",
    "IssuerType",
    "SecurityType",
    "SecuritySubType",
    "ShareType",
    "MthRetFlg",
    "MthDelFlg",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Inventory and QA CRSP/FF3 files.")
    result.add_argument("--data-dir")
    result.add_argument("--output-json", default="reports/phase0_inventory.json")
    result.add_argument("--output-md", default="reports/phase0_inventory.md")
    result.add_argument("--chunksize", type=int, default=500_000)
    return result


def inspect_gzip_csv(
    path: Path,
    date_column: str,
    key_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    chunksize: int,
) -> dict:
    with path.open("rb") as stream:
        magic = stream.read(2).hex()
    is_gzip = magic == "1f8b"
    if not is_gzip:
        raise ValueError(f"{path.name} does not have gzip magic bytes")

    header = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
    missing_required = sorted(
        set([date_column, *key_columns, *categorical_columns, *numeric_columns])
        - set(header)
    )
    if missing_required:
        raise ValueError(f"{path.name} missing columns: {missing_required}")

    row_count = 0
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    invalid_date_count = 0
    missing_counts = Counter({column: 0 for column in header})
    category_counts = {column: Counter() for column in categorical_columns}
    numeric_summary = {
        column: {"invalid_nonmissing": 0, "valid": 0, "min": None, "max": None}
        for column in numeric_columns
    }
    key_hashes: list[np.ndarray] = []
    row_hashes: list[np.ndarray] = []
    missing_key_rows = 0

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        chunksize=chunksize,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    ):
        row_count += len(chunk)
        for column in header:
            missing_counts[column] += int(chunk[column].isna().sum())

        dates = pd.to_datetime(chunk[date_column], errors="coerce")
        invalid_date_count += int(dates.isna().sum())
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            chunk_min = valid_dates.min()
            chunk_max = valid_dates.max()
            min_date = chunk_min if min_date is None else min(min_date, chunk_min)
            max_date = chunk_max if max_date is None else max(max_date, chunk_max)

        for column in categorical_columns:
            category_counts[column].update(
                chunk[column].fillna("<MISSING>").astype(str).tolist()
            )

        for column in numeric_columns:
            raw = chunk[column]
            numeric = pd.to_numeric(raw, errors="coerce")
            invalid = raw.notna() & numeric.isna()
            valid = numeric.dropna()
            numeric_summary[column]["invalid_nonmissing"] += int(invalid.sum())
            numeric_summary[column]["valid"] += int(valid.size)
            if not valid.empty:
                value_min = float(valid.min())
                value_max = float(valid.max())
                previous_min = numeric_summary[column]["min"]
                previous_max = numeric_summary[column]["max"]
                numeric_summary[column]["min"] = (
                    value_min if previous_min is None else min(previous_min, value_min)
                )
                numeric_summary[column]["max"] = (
                    value_max if previous_max is None else max(previous_max, value_max)
                )

        key_frame = chunk[key_columns]
        missing_key_rows += int(key_frame.isna().any(axis=1).sum())
        hashes = pd.util.hash_pandas_object(
            key_frame.astype("string"), index=False, categorize=True
        ).to_numpy(dtype=np.uint64, copy=True)
        key_hashes.append(hashes)
        row_hashes.append(
            pd.util.hash_pandas_object(
                chunk[header].astype("string"), index=False, categorize=True
            ).to_numpy(dtype=np.uint64, copy=True)
        )

    hashes = np.concatenate(key_hashes) if key_hashes else np.array([], dtype=np.uint64)
    rows = np.concatenate(row_hashes) if row_hashes else np.array([], dtype=np.uint64)
    if hashes.size:
        order = np.argsort(hashes, kind="stable")
        hashes = hashes[order]
        rows = rows[order]
        run_starts = np.r_[True, hashes[1:] != hashes[:-1]]
        starts = np.flatnonzero(run_starts)
        run_lengths = np.diff(np.r_[starts, hashes.size])
        duplicate_key_count = int(np.count_nonzero(run_lengths > 1))
        duplicate_row_count = int(np.sum(run_lengths[run_lengths > 1] - 1))
        exact_duplicate_rows = 0
        conflicting_duplicate_keys = 0
        for start, length in zip(starts, run_lengths):
            if length <= 1:
                continue
            if np.unique(rows[start : start + length]).size == 1:
                exact_duplicate_rows += int(length - 1)
            else:
                conflicting_duplicate_keys += 1
    else:
        duplicate_key_count = 0
        duplicate_row_count = 0
        exact_duplicate_rows = 0
        conflicting_duplicate_keys = 0

    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "gzip_magic": magic,
        "compression_ok": is_gzip,
        "columns": header,
        "row_count": int(row_count),
        "date_column": date_column,
        "min_date": None if min_date is None else min_date.date().isoformat(),
        "max_date": None if max_date is None else max_date.date().isoformat(),
        "invalid_date_count": int(invalid_date_count),
        "key_columns": key_columns,
        "missing_key_rows": int(missing_key_rows),
        "duplicate_key_count": duplicate_key_count,
        "duplicate_row_count_above_first": duplicate_row_count,
        "exact_duplicate_rows_above_first": exact_duplicate_rows,
        "conflicting_duplicate_keys": conflicting_duplicate_keys,
        "duplicate_method": "64-bit deterministic hashes of composite key and full row",
        "missing_counts": dict(missing_counts),
        "numeric_summary": numeric_summary,
        "category_counts": {
            column: dict(counter.most_common())
            for column, counter in category_counts.items()
        },
    }


def inspect_ff3(path: Path) -> dict:
    frame = load_ff3(path)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "members": ["F-F_Research_Data_Factors_daily.csv"],
        "row_count": int(len(frame)),
        "min_date": frame["date"].min().date().isoformat(),
        "max_date": frame["date"].max().date().isoformat(),
        "duplicate_dates": int(frame["date"].duplicated().sum()),
        "missing_counts": {
            column: int(frame[column].isna().sum()) for column in frame.columns
        },
        "factor_units_in_source": "percent",
        "factor_units_after_load_ff3": "decimal (divided by 100)",
    }


def build_markdown(report: dict) -> str:
    csv_files = report["crsp_files"]
    lines = [
        "# Phase 0 — Inventory and QA",
        "",
        "원자료는 읽기 전용으로 스캔했으며 수정하거나 이름을 바꾸지 않았다.",
        "복합키 중복은 64-bit 결정론적 해시로 확인했다(충돌 가능성은 극히 작지만 0은 아님).",
        "",
        "## 파일 요약",
        "",
        "| 파일 | 행 수 | 기간 | 키 중복 행 | 충돌 키 | gzip |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for item in csv_files:
        lines.append(
            f"| `{item['file']}` | {item['row_count']:,} | "
            f"{item['min_date']} ~ {item['max_date']} | "
            f"{item['duplicate_row_count_above_first']:,} | "
            f"{item['conflicting_duplicate_keys']:,} | "
            f"{'정상' if item['compression_ok'] else '오류'} |"
        )
    ff3 = report["ff3"]
    lines.extend(
        [
            f"| `{ff3['file']}` | {ff3['row_count']:,} | "
            f"{ff3['min_date']} ~ {ff3['max_date']} | "
            f"{ff3['duplicate_dates']:,} | 0 | ZIP |",
            "",
            "## 스키마",
            "",
            f"- Daily 스키마 일치: **{report['daily_schema_consistent']}**",
            f"- 이중 확장자 파일: {', '.join(report['double_gzip_names']) or '없음'}",
            "- 이중 `.gz`는 파일명 문제이며 gzip magic bytes는 별도로 검증했다.",
            "",
            "## 분석상 주의사항",
            "",
            "- FF3 원자료는 퍼센트 단위이며 `load_ff3`에서 100으로 나눈다.",
            "- `NA`는 `DlyRetMissFlg`/`MthRetFlg`의 실제 코드일 수 있어 일반 결측 토큰으로 해석하지 않았다.",
            "- 기본 거래소 코드는 표본에서 확인된 NYSE=`N`, AMEX=`A`, Nasdaq=`Q`를 사용한다.",
            "- 원자료 파일명은 보존한다. 파이프라인은 `.gz.gz`도 gzip 내용 기준으로 읽는다.",
            "- 완전 동일한 복제행은 첫 행만 유지한다. 같은 키의 필드 값이 충돌하면 파이프라인은 실패한다.",
            "",
            "## 범주형 코드 빈도",
            "",
        ]
    )
    for item in csv_files:
        lines.append(f"### {item['file']}")
        lines.append("")
        for column, counts in item["category_counts"].items():
            rendered = ", ".join(f"`{key}`={value:,}" for key, value in counts.items())
            lines.append(f"- {column}: {rendered}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parser().parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    required = [*DAILY_FILES, MONTHLY_FILE, FF3_FILE]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing expected files: {missing}")

    daily_reports = []
    for filename in DAILY_FILES:
        print(f"Scanning {filename}...", flush=True)
        daily_reports.append(
            inspect_gzip_csv(
                data_dir / filename,
                date_column="DlyCalDt",
                key_columns=["PERMNO", "DlyCalDt"],
                categorical_columns=DAILY_CATEGORIES,
                numeric_columns=["DlyRet"],
                chunksize=args.chunksize,
            )
        )

    print(f"Scanning {MONTHLY_FILE}...", flush=True)
    monthly_report = inspect_gzip_csv(
        data_dir / MONTHLY_FILE,
        date_column="MthCalDt",
        key_columns=["PERMNO", "MthCalDt"],
        categorical_columns=MONTHLY_CATEGORIES,
        numeric_columns=["MthRet", "MthCap", "MthPrevCap"],
        chunksize=args.chunksize,
    )
    print(f"Scanning {FF3_FILE}...", flush=True)
    ff3_report = inspect_ff3(data_dir / FF3_FILE)

    schemas = [tuple(item["columns"]) for item in daily_reports]
    report = {
        "data_dir": str(data_dir),
        "raw_files_modified": False,
        "crsp_files": [*daily_reports, monthly_report],
        "ff3": ff3_report,
        "daily_schema_consistent": len(set(schemas)) == 1,
        "double_gzip_names": [name for name in DAILY_FILES if name.endswith(".gz.gz")],
    }
    write_json(report, args.output_json)
    output_md = ensure_parent(args.output_md)
    output_md.write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
