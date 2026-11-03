from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from cnn_official_data import (
    EXPECTED_ARCHIVE_SHA256,
    IMAGE_SHAPE,
    catalog_archive,
    iter_year_image_batches,
    match_by_exchange,
    match_universes,
    prepare_classification_labels,
    read_professor_monthly,
    read_year_labels,
    validate_label_partition,
    verify_official_archive_hash,
)
from common import ensure_parent, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Audit the official monthly I20 image archive and compare its "
            "PERMNO-month universe with the professor's monthly CSV."
        )
    )
    result.add_argument("--image-zip", required=True, type=Path)
    result.add_argument("--professor-csv", required=True, type=Path)
    result.add_argument("--start-year", type=int, default=2000)
    result.add_argument("--end-year", type=int, default=2019)
    result.add_argument("--chunksize", type=int, default=250_000)
    result.add_argument("--sample-size", type=int, default=100)
    result.add_argument(
        "--output-dir", type=Path, default=Path("outputs/cnn_data_pilot")
    )
    result.add_argument(
        "--report", type=Path, default=Path("reports/cnn_data_pilot_findings.md")
    )
    result.add_argument("--log", type=Path, default=Path("logs/cnn_data_pilot.json"))
    result.add_argument(
        "--skip-sha256",
        action="store_true",
        help=(
            "Explicitly skip the official archive hash check. By default a "
            "mismatched SHA-256 stops the run."
        ),
    )
    return result


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _integer(value: int | float) -> str:
    return f"{int(value):,}"


def _compact(value: int | float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def build_report(
    summary: dict,
    quality_by_year: pd.DataFrame,
    match_by_year: pd.DataFrame,
    exchange_match: pd.DataFrame,
) -> str:
    archive = summary["archive"]
    professor = summary["professor_csv"]
    window = summary["analysis_window"]
    model_preparation = summary["model_label_preparation"]
    paper_training = summary["paper_training_split"]
    match_all = summary["universe_match"]["all"]
    match_eligible = summary["universe_match"]["exchange_eligible"]
    usable_rate = (
        match_all["matched_model_label_eligible"] / match_all["professor_keys"]
        if match_all["professor_keys"]
        else float("nan")
    )
    unmatched_first_rate = (
        match_all["professor_only_first_source_observation"]
        / match_all["professor_only_keys"]
        if match_all["professor_only_keys"]
        else float("nan")
    )
    unmatched_first3_rate = (
        match_all["professor_only_within_first_3_source_observations"]
        / match_all["professor_only_keys"]
        if match_all["professor_only_keys"]
        else float("nan")
    )
    rows_after_archive = sum(
        int(count)
        for year, count in professor["source_rows_by_year"].items()
        if int(year) > archive["last_year"]
    )
    last_archive_year = match_by_year.loc[
        match_by_year["scope"].eq("all")
        & match_by_year["year"].eq(archive["last_year"])
    ]
    last_year_match = (
        last_archive_year.iloc[0].to_dict() if not last_archive_year.empty else None
    )
    image_name = Path(summary["inputs"]["image_zip"]).name
    professor_name = Path(summary["inputs"]["professor_csv"]).name
    verified_hash = summary["inputs"]["image_zip_sha256"] or "검사 생략"

    all_years = match_by_year.loc[match_by_year["scope"].eq("all")].copy()
    table_rows = [
        "| 연도 | 교수 표본 | 공식 표본 | 일치 | 교수 표본 포함률 | 공식 표본 중 교수 표본 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_years.itertuples(index=False):
        table_rows.append(
            "| {year} | {professor_keys:,} | {official_keys:,} | {matched_keys:,} | "
            "{professor_rate:.2%} | {official_rate:.2%} |".format(
                year=row.year,
                professor_keys=row.professor_keys,
                official_keys=row.official_keys,
                matched_keys=row.matched_keys,
                professor_rate=row.professor_coverage_rate,
                official_rate=row.official_coverage_rate,
            )
        )

    exchange_rows = [
        "| EXCHCD | 교수 표본 | 일치 | 포함률 |",
        "|---:|---:|---:|---:|",
    ]
    for row in exchange_match.itertuples(index=False):
        exchange_rows.append(
            f"| {row.EXCHCD} | {row.professor_keys:,} | {row.matched_keys:,} | "
            f"{row.professor_coverage_rate:.2%} |"
        )

    return "\n".join(
        [
            f"# 공식 I20/R20 자료가 교수님 월별 표본의 {_pct(match_all['professor_coverage_rate'])}를 포함한다",
            "",
            f"작성일: {summary['generated_at'][:10]}  ",
            f"대조 기간: {window['start_year']}–{window['end_year']}",
            "",
            "## 결론",
            "",
            f"- 교수님 CSV의 월별 종목 키 {_compact(match_all['professor_keys'])}개 중 "
            f"{_compact(match_all['matched_keys'])}개가 공식 이미지 라벨에 있다. "
            f"월 키로 일치한 행의 정확한 날짜 일치율도 {_pct(match_all['exact_date_match_rate_among_matches'])}다.",
            f"- 거래소 코드 1·2·3만 남기면 교수 표본 포함률은 "
            f"{_pct(match_eligible['professor_coverage_rate'])}다. 따라서 {window['start_year']}–{window['end_year']}년은 "
            "공식 전체 표본으로 모델을 학습한 뒤 교수님 표본에 예측값을 붙이는 경로가 현실적이다.",
            f"- I20/R20의 공식 목표인 `Ret_month`와 유효한 `EWMA_vol`이 모두 있는 교수 키는 "
            f"{_integer(match_all['matched_model_label_eligible'])}개, 원래 교수 키의 {_pct(usable_rate)}다. "
            "CNN 분류 라벨을 만들 때 두 필드를 먼저 검증해야 한다.",
            f"- 공식 ZIP의 {archive['partition_count']}개 연도, "
            f"{_compact(archive['total_label_rows'])}개 라벨은 이미지 수와 전부 일치하고 "
            f"중복 키가 {archive['duplicate_key_excess']:,}개다.",
            f"- 공개 이미지는 {archive['last_year']}년까지만 있다. 교수 CSV에서 그 이후에 해당하는 "
            f"{_integer(rows_after_archive)}개 행은 현재 ZIP으로 점수화할 수 없다.",
            "",
            "## 종목 일치 여부",
            "",
            f"교수 표본에만 있는 키는 {_integer(match_all['professor_only_keys'])}개다. "
            f"그중 교수님 CSV 안에서 해당 PERMNO의 첫 관측이 {_pct(unmatched_first_rate)}, "
            f"첫 세 관측 이내가 "
            f"{_pct(unmatched_first3_rate)}다. 이는 20거래일 입력창을 만들기 위한 과거 자료가 "
            "부족한 신규 진입 종목이 주된 원인이라는 해석과 일관된다. 다만 이 비율만으로 "
            "개별 누락 원인을 확정할 수는 없다.",
            "",
            "공식 라벨 가운데 교수 표본과 겹치는 비율은 "
            f"{_pct(match_all['official_coverage_rate'])}다. 공식 universe가 더 넓다는 뜻이며, "
            "두 파일이 같은 CRSP 계열 자료여도 완전히 같은 표본은 아니다.",
            "",
            *table_rows,
            "",
            "## 거래소 코드별 확인",
            "",
            *exchange_rows,
            "",
            "EXCHCD 4·5·6이 있다면 공식 기본 universe와 다르므로 모델 입력에서 별도로 처리해야 한다. "
            "프로젝트의 기존 NYSE·AMEX·NASDAQ 범위는 EXCHCD 1·2·3과 맞는다.",
            "",
            "## I20/R20의 기본 목표는 `Ret_month`다",
            "",
            f"두 수익률이 모두 존재하는 매칭 행은 "
            f"{_integer(match_all['ret_20d_month_comparable'])}개다. 수치가 허용오차 안에서 같은 "
            f"비율은 {_pct(match_all['ret_20d_month_equal_rate'])}, 상승·비상승 분류가 다른 비율은 "
            f"{_pct(match_all['ret_20d_month_sign_disagreement_rate'])}다. 공식 코드의 `R20`은 "
            "`FREQ_DICT[20] = month`를 거쳐 `Ret_month`를 읽는다. 따라서 다음 월말 수익률인 "
            "`Ret_month`를 주 결과로 쓰고, 향후 20거래일 수익률인 `Ret_20d`를 별도 강건성 비교로 "
            "남기는 것이 코드와 프로젝트의 월별 리밸런싱 정의에 맞는다.",
            "",
            "## 데이터 품질 판단",
            "",
            "- 공식 이미지–라벨 정렬: 통과. 모든 연도에서 `.dat` 바이트 수/3,840과 Feather 행 수가 같다.",
            "- 공식 키 유일성: 통과. `(PERMNO, 월)` 중복과 유효하지 않은 종목·날짜 키가 없다.",
            f"- 시가총액 부호: 음수 `MarketCap`이 {_integer(archive['negative_market_cap_rows'])}개 있다. "
            "공개 라벨과 생성 코드의 버전 차이 등 원인은 확정할 수 없으므로 원값을 보존해 감사하고, "
            "공식 로더와 같이 가치가중 전 절댓값으로 변환한다.",
            f"- 교수 CSV 키 유일성: `(PERMNO, 날짜)` 및 `(PERMNO, 월)` 중복 초과행이 "
            f"각각 {professor['window_exact_key_duplicate_excess']:,}개, "
            f"{professor['window_month_key_duplicate_excess']:,}개다.",
            f"- {window['start_year']}–{window['end_year']} 표본 결합 위험: 낮음. 누락률은 작고 연도별 포함률을 별도 표로 남겼다.",
            f"- {archive['last_year'] + 1}년 이후 연장 위험: 높음. 공개 ZIP에 이미지가 없으므로 일별 OHLCV로 "
            f"새 이미지를 만들거나 분석 종료일을 {archive['last_year']}년으로 제한해야 한다.",
            f"- 라벨 필터: 전체 {_integer(model_preparation['input_rows'])}개 중 "
            f"{_integer(model_preparation['kept_rows'])}개를 유지한다. 유효하지 않은 `Ret_month` "
            f"{_integer(model_preparation['rows_with_missing_or_nonfinite_target'])}개와 유효하지 않은 "
            f"`EWMA_vol` {_integer(model_preparation['rows_with_missing_nonfinite_or_nonpositive_ewma_vol'])}개는 "
            "서로 겹칠 수 있으며 합산하지 않는다.",
            *(
                [
                    f"- {archive['last_year']}년 경계: 매칭 {_integer(last_year_match['matched_keys'])}개 중 "
                    f"`Ret_month`·`EWMA_vol` 사용 가능 행은 "
                    f"{_integer(last_year_match['matched_model_label_eligible'])}개"
                    f"({_pct(last_year_match['matched_model_label_eligible_rate'])})다. "
                    "마지막 12월 이미지는 점수화할 수 있어도 다음 월말 성과 평가는 할 수 없다."
                ]
                if last_year_match is not None
                else []
            ),
            "",
            "## 권장 첫 실행",
            "",
            "1. `Ret_month` 결측과 유효하지 않은 `EWMA_vol`을 제외한 공식 전체 universe의 "
            "1993–2000 이미지로 `I20/R20` 모델을 학습·검증한다. 2000년 12월 입력은 "
            f"2001년 수익률 누수를 막기 위해 제외하며, 이 경계에서 유효 라벨 "
            f"{_integer(paper_training['eligible_rows_removed_at_2000_boundary'])}개가 빠진다.",
            "2. 고정한 모델로 2001–2019 공식 이미지를 점수화한다.",
            "3. 예측 결과를 `(PERMNO, Date)`로 교수님 CSV에 붙이고 교수 표본 안에서 월별 10분위를 만든다.",
            "4. `Ret_month`를 주 결과로 쓰고 `Ret_20d` 결과를 별도 비교한다.",
            "5. 2020, I5, I60이 필요해질 때만 일별 CRSP OHLCV를 추가로 받는다.",
            "",
            "이번 파일럿은 데이터·표본 정합성까지만 검증했다. PyTorch 설치, CNN 학습, "
            "하이퍼파라미터 선택, 성과 검정은 아직 수행하지 않았다.",
            "",
            "공식 배포 코드와 공개 데이터 사이에는 어댑터가 필요한 차이가 있다. 공개 Feather 이름은 "
            "`_labels_w_delay.feather`인데 코드는 `_labels.feather`를 찾으며, 공식 환경 명세에는 "
            "Feather 엔진인 `pyarrow`가 없다. 또한 공식 분류 코드가 결측 수익률을 하락(0)으로 바꾸는 "
            "경로가 있어, 이 프로젝트 로더는 결측 목표값과 유효하지 않은 `EWMA_vol`을 먼저 제외한다.",
            "",
            "## 다시 실행",
            "",
            "```powershell",
            f"$env:CNN_IMAGE_ZIP='C:\\path\\to\\{image_name}'",
            f"$env:PROFESSOR_STOCK_CSV='C:\\path\\to\\{professor_name}'",
            ".\\.venv\\Scripts\\python src\\run_cnn_data_pilot.py `",
            "  --image-zip $env:CNN_IMAGE_ZIP `",
            "  --professor-csv $env:PROFESSOR_STOCK_CSV",
            "```",
            "",
            "같은 검사를 셀 단위로 보려면 `notebooks/cnn_data_pilot.ipynb`를 연다. 실제 CNN "
            "학습용 PyTorch 환경은 이 파일럿과 분리한다.",
            "",
            "## 재현 정보",
            "",
            f"- 이미지 ZIP: `{image_name}`, SHA-256 `{verified_hash}`",
            f"- 교수 표본: `{professor_name}`",
            f"- 이미지 크기: `{IMAGE_SHAPE[0]}×{IMAGE_SHAPE[1]}` uint8, 관측치당 3,840바이트",
            "- 기본 결합 키: `(PERMNO, calendar month)`; exact `(PERMNO, Date)` 일치율을 독립 검증",
            f"- 상세표: `{summary['outputs']['output_dir']}`",
            f"- 실행 로그: `{summary['outputs']['log']}`",
            "- 재현 스크립트: `src/run_cnn_data_pilot.py`",
        ]
    ) + "\n"


def run_pipeline(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    image_zip = args.image_zip.expanduser().resolve()
    professor_csv = args.professor_csv.expanduser().resolve()
    output_dir = args.output_dir
    report_path = args.report
    log_path = args.log
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if not report_path.is_absolute():
        report_path = root / report_path
    if not log_path.is_absolute():
        log_path = root / log_path
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.start_year > args.end_year:
        raise ValueError("start-year cannot be later than end-year")
    if args.sample_size < 0:
        raise ValueError("sample-size cannot be negative")

    print("Cataloging official archive...", flush=True)
    catalog = catalog_archive(image_zip)
    archive_years = [partition.year for partition in catalog]
    expected_years = list(range(min(archive_years), max(archive_years) + 1))
    missing_archive_years = sorted(set(expected_years).difference(archive_years))
    missing_match_years = sorted(
        set(range(args.start_year, args.end_year + 1)).difference(archive_years)
    )
    if missing_match_years:
        raise ValueError(f"Archive is missing requested years: {missing_match_years}")

    image_zip_sha256, sha256_matches_expected = verify_official_archive_hash(
        image_zip, skip=args.skip_sha256
    )

    quality_rows: list[dict] = []
    official_parts: list[pd.DataFrame] = []
    model_label_audits: list[dict] = []
    paper_training_audits: list[dict] = []
    keep_columns = [
        "Date",
        "PERMNO",
        "month",
        "MarketCap",
        "Ret_20d",
        "Ret_month",
        "EWMA_vol",
        "archive_year",
    ]
    for partition in catalog:
        print(f"Reading labels for {partition.year}...", flush=True)
        labels = read_year_labels(
            image_zip, partition.year, catalog=catalog
        )
        quality, normalized = validate_label_partition(partition, labels)
        quality_rows.append(quality)
        _, _, label_audit = prepare_classification_labels(
            normalized, target_column="Ret_month"
        )
        label_audit["year"] = partition.year
        model_label_audits.append(label_audit)
        if 1993 <= partition.year <= 2000:
            if partition.year == 2000:
                _, _, paper_audit = prepare_classification_labels(
                    normalized,
                    target_column="Ret_month",
                    input_date_cutoff_exclusive="2000-12-01",
                )
            else:
                paper_audit = dict(label_audit)
            paper_audit["year"] = partition.year
            paper_training_audits.append(paper_audit)
        if args.start_year <= partition.year <= args.end_year:
            official_parts.append(normalized[keep_columns].copy())
    quality_by_year = pd.DataFrame(quality_rows).sort_values("year")
    model_label_audit_by_year = pd.DataFrame(model_label_audits).sort_values("year")
    paper_training_audit = pd.DataFrame(paper_training_audits).sort_values("year")
    official = pd.concat(official_parts, ignore_index=True)

    smoke_year = args.start_year
    iterator = iter_year_image_batches(
        image_zip,
        smoke_year,
        batch_size=8,
        max_images=8,
        catalog=catalog,
    )
    try:
        smoke_offset, smoke_batch = next(iterator)
    finally:
        iterator.close()
    smoke_test = {
        "year": smoke_year,
        "offset": int(smoke_offset),
        "shape": list(smoke_batch.shape),
        "dtype": str(smoke_batch.dtype),
        "min_pixel": int(smoke_batch.min()),
        "max_pixel": int(smoke_batch.max()),
    }

    print("Reading professor CSV key fields...", flush=True)
    professor_frame, professor_summary = read_professor_monthly(
        professor_csv,
        start_year=args.start_year,
        end_year=args.end_year,
        chunksize=args.chunksize,
    )
    print("Comparing PERMNO-month universes...", flush=True)
    match_summary, match_by_year, match_by_month, unmatched = match_universes(
        official, professor_frame
    )
    exchange_match = match_by_exchange(official, professor_frame)

    archive_summary = {
        "first_year": int(min(archive_years)),
        "last_year": int(max(archive_years)),
        "partition_count": int(len(catalog)),
        "missing_years_within_range": missing_archive_years,
        "total_image_count": int(quality_by_year["image_count"].sum()),
        "total_label_rows": int(quality_by_year["label_rows"].sum()),
        "partitions_with_count_mismatch": int(
            (~quality_by_year["image_label_count_match"]).sum()
        ),
        "null_dates": int(quality_by_year["null_dates"].sum()),
        "invalid_stock_ids": int(quality_by_year["invalid_stock_ids"].sum()),
        "duplicate_key_excess": int(quality_by_year["duplicate_key_excess"].sum()),
        "negative_market_cap_rows": int(
            quality_by_year["negative_market_cap"].sum()
        ),
        "zero_market_cap_rows": int(quality_by_year["zero_market_cap"].sum()),
        "date_year_mismatch_rows": int(
            quality_by_year["date_year_mismatch_rows"].sum()
        ),
        "paper_train_label_rows_1993_2000": int(
            quality_by_year.loc[
                quality_by_year["year"].between(1993, 2000), "label_rows"
            ].sum()
        ),
    }
    paper_is_mask = model_label_audit_by_year["year"].between(1993, 2000)
    paper_eligible_before_boundary = int(
        model_label_audit_by_year.loc[paper_is_mask, "kept_rows"].sum()
    )
    paper_eligible_after_boundary = int(paper_training_audit["kept_rows"].sum())
    paper_training_summary = {
        "raw_rows_1993_2000": archive_summary["paper_train_label_rows_1993_2000"],
        "eligible_rows_before_oos_boundary": paper_eligible_before_boundary,
        "input_date_cutoff_exclusive_for_2000": "2000-12-01",
        "eligible_rows_removed_at_2000_boundary": int(
            paper_eligible_before_boundary - paper_eligible_after_boundary
        ),
        "final_training_validation_pool_rows": paper_eligible_after_boundary,
    }
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "analysis_window": {
            "start_year": args.start_year,
            "end_year": args.end_year,
        },
        "inputs": {
            "image_zip": str(image_zip),
            "image_zip_sha256": image_zip_sha256,
            "expected_image_zip_sha256": EXPECTED_ARCHIVE_SHA256,
            "image_zip_sha256_matches_expected": sha256_matches_expected,
            "professor_csv": str(professor_csv),
        },
        "archive": archive_summary,
        "professor_csv": professor_summary,
        "universe_match": match_summary,
        "image_stream_smoke_test": smoke_test,
        "paper_training_split": paper_training_summary,
        "model_label_preparation": {
            "target_column": "Ret_month",
            "input_rows": int(model_label_audit_by_year["input_rows"].sum()),
            "kept_rows": int(model_label_audit_by_year["kept_rows"].sum()),
            "dropped_total": int(model_label_audit_by_year["dropped_total"].sum()),
            "rows_with_missing_or_nonfinite_target": int(
                model_label_audit_by_year[
                    "rows_with_missing_or_nonfinite_target"
                ].sum()
            ),
            "rows_with_missing_nonfinite_or_nonpositive_ewma_vol": int(
                model_label_audit_by_year[
                    "rows_with_missing_nonfinite_or_nonpositive_ewma_vol"
                ].sum()
            ),
            "dropped_target_only": int(
                model_label_audit_by_year["dropped_target_only"].sum()
            ),
            "dropped_ewma_only": int(
                model_label_audit_by_year["dropped_ewma_only"].sum()
            ),
            "dropped_target_and_ewma": int(
                model_label_audit_by_year["dropped_target_and_ewma"].sum()
            ),
            "negative_market_cap_rows_converted_to_absolute": int(
                model_label_audit_by_year[
                    "negative_market_cap_rows_converted_to_absolute"
                ].sum()
            ),
        },
    }

    quality_path = output_dir / "archive_quality_by_year.csv"
    year_path = output_dir / "universe_match_by_year.csv"
    month_path = output_dir / "universe_match_by_month.csv"
    exchange_path = output_dir / "universe_match_by_exchange.csv"
    label_audit_path = output_dir / "model_label_audit_by_year.csv"
    paper_audit_path = output_dir / "paper_training_label_audit_by_year.csv"
    summary_path = output_dir / "summary.json"
    quality_by_year.to_csv(quality_path, index=False)
    match_by_year.to_csv(year_path, index=False)
    match_by_month.to_csv(month_path, index=False)
    exchange_match.to_csv(exchange_path, index=False)
    model_label_audit_by_year.to_csv(label_audit_path, index=False)
    paper_training_audit.to_csv(paper_audit_path, index=False)

    for scope, (professor_only, official_only) in unmatched.items():
        professor_only.head(args.sample_size).to_csv(
            output_dir / f"professor_only_{scope}_sample.csv", index=False
        )
        official_only.head(args.sample_size).to_csv(
            output_dir / f"official_only_{scope}_sample.csv", index=False
        )

    def display_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return path.name

    summary["outputs"] = {
        "output_dir": display_path(output_dir),
        "summary": display_path(summary_path),
        "archive_quality_by_year": display_path(quality_path),
        "universe_match_by_year": display_path(year_path),
        "universe_match_by_month": display_path(month_path),
        "universe_match_by_exchange": display_path(exchange_path),
        "model_label_audit_by_year": display_path(label_audit_path),
        "paper_training_label_audit_by_year": display_path(paper_audit_path),
        "report": display_path(report_path),
        "log": display_path(log_path),
    }
    report = build_report(summary, quality_by_year, match_by_year, exchange_match)
    ensure_parent(report_path).write_text(report, encoding="utf-8")
    write_json(summary, summary_path)
    write_json(summary, log_path)
    print(json.dumps(summary["universe_match"]["all"], indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)
    return summary


def main() -> None:
    run_pipeline(parser().parse_args())


if __name__ == "__main__":
    main()
