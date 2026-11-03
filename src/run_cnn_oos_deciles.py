from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT if (ROOT / "handoff").is_dir() else Path(__file__).resolve().parents[2]
HANDOFF_SRC = PROJECT_ROOT / "handoff" / "ivol_combination" / "src"
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(HANDOFF_SRC) not in sys.path:
    sys.path.insert(0, str(HANDOFF_SRC))

from cnn_official_data import (  # noqa: E402
    EXPECTED_ARCHIVE_SHA256,
    IMAGE_SHAPE,
    catalog_archive,
    iter_filtered_year_image_batches,
    read_year_labels,
    validate_label_partition,
    verify_official_archive_hash,
)
from performance_metrics import (  # noqa: E402
    add_long_short,
    summarize_portfolios,
)
from run_cnn_epoch_benchmark import I20CNN  # noqa: E402
from shared_contract import (  # noqa: E402
    build_adj_price_outcomes,
    load_professor_prices,
    official_formation_universe,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Freeze the trained I20 CNN ensemble, score the professor-universe "
            "OOS stock-months, and evaluate next-month return deciles."
        )
    )
    result.add_argument("--image-zip", type=Path, required=True)
    result.add_argument("--professor-csv", type=Path, required=True)
    result.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "cnn_overnight" / "i20_ret_month_1993_2000",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "cnn_oos_deciles_2001_2019",
    )
    result.add_argument("--start-year", type=int, default=2001)
    result.add_argument("--end-year", type=int, default=2019)
    result.add_argument("--batch-size", type=int, default=512)
    result.add_argument("--progress-every", type=int, default=25_000)
    result.add_argument("--skip-sha256", action="store_true")
    return result


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def assign_cnn_deciles(
    frame: pd.DataFrame,
    *,
    month_column: str = "formation_month",
    score_column: str = "cnn_prob_up",
    id_column: str = "PERMNO",
) -> pd.Series:
    """Deterministic equal-count deciles; Q10 has the highest CNN score."""

    required = {month_column, score_column, id_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot assign CNN deciles; missing columns: {missing}")
    result = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for month, index in frame.groupby(month_column, sort=True).groups.items():
        ordered = frame.loc[index].sort_values(
            [score_column, id_column], kind="mergesort"
        )
        n_rows = len(ordered)
        if n_rows < 10:
            raise ValueError(f"Month {month} has only {n_rows} scored stocks")
        bins = (np.arange(n_rows, dtype=np.int64) * 10 // n_rows) + 1
        result.loc[ordered.index] = bins
    return result


def month_rank_percentile(values: pd.Series) -> pd.Series:
    if len(values) == 1:
        return pd.Series([0.5], index=values.index, dtype=float)
    ranks = values.rank(method="average", ascending=True)
    return (ranks - 1.0) / (len(values) - 1.0)


def ensemble_up_probabilities(
    models: Sequence[torch.nn.Module], inputs: torch.Tensor
) -> torch.Tensor:
    if not models:
        raise ValueError("At least one CNN model is required")
    probability_sum = torch.zeros(len(inputs), device=inputs.device, dtype=torch.float32)
    with torch.inference_mode():
        for model in models:
            logits = model(inputs)
            if logits.shape != (len(inputs), 2):
                raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")
            probability_sum.add_(torch.softmax(logits, dim=1)[:, 1])
    return probability_sum / float(len(models))


def _load_summary(checkpoint_dir: Path) -> dict[str, Any]:
    summary_path = checkpoint_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"CNN training summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("CNN training summary is not complete")
    if summary.get("archive", {}).get("sha256") != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Training summary archive hash differs from the official archive")
    normalization = summary.get("normalization", {})
    mean = float(normalization.get("mean", math.nan))
    std = float(normalization.get("std", math.nan))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("Training summary has invalid frozen normalization")
    if len(summary.get("models", [])) != 5:
        raise ValueError("Expected exactly five completed ensemble models")
    return summary


def _load_models(
    checkpoint_dir: Path, summary: dict[str, Any], device: torch.device
) -> tuple[list[I20CNN], list[dict[str, Any]]]:
    models: list[I20CNN] = []
    audit: list[dict[str, Any]] = []
    for item in summary["models"]:
        checkpoint = Path(item["best_checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = checkpoint_dir / checkpoint
        if not checkpoint.is_file():
            fallback = checkpoint_dir / f"model_{item['model_index']:02d}_seed{item['model_seed']}" / "best.pt"
            checkpoint = fallback
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = I20CNN()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
        models.append(model)
        audit.append(
            {
                "model_index": int(item["model_index"]),
                "model_seed": int(item["model_seed"]),
                "best_epoch_number": int(item["best_epoch_number"]),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    return models, audit


def _professor_membership(
    normalized: pd.DataFrame, formation: pd.DataFrame
) -> np.ndarray:
    official_keys = pd.MultiIndex.from_frame(
        normalized[["PERMNO", "month"]].astype({"PERMNO": "int64", "month": str})
    )
    professor_keys = pd.MultiIndex.from_frame(
        formation[["PERMNO", "formation_month"]].rename(
            columns={"formation_month": "month"}
        )
    )
    return np.asarray(official_keys.isin(professor_keys), dtype=bool)


def score_year(
    *,
    image_zip: Path,
    year: int,
    catalog: Sequence[Any],
    formation: pd.DataFrame,
    models: Sequence[torch.nn.Module],
    device: torch.device,
    mean: float,
    std: float,
    batch_size: int,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = read_year_labels(image_zip, year, catalog=catalog)
    profile, normalized = validate_label_partition(
        next(item for item in catalog if item.year == year), labels
    )
    keep_mask = _professor_membership(normalized, formation)
    selected_rows = np.flatnonzero(keep_mask)
    selected = normalized.iloc[selected_rows][
        ["PERMNO", "Date", "month", "MarketCap"]
    ].copy()
    selected = selected.rename(
        columns={
            "Date": "official_date",
            "month": "formation_month",
            "MarketCap": "formation_market_cap",
        }
    )
    selected["PERMNO"] = selected["PERMNO"].astype("int64")
    selected["formation_market_cap"] = np.abs(
        pd.to_numeric(selected["formation_market_cap"], errors="coerce")
    )

    probabilities: list[np.ndarray] = []
    emitted_rows: list[np.ndarray] = []
    processed = 0
    started = time.perf_counter()
    for source_rows, images in iter_filtered_year_image_batches(
        image_zip,
        year,
        keep_mask,
        batch_size=batch_size,
        catalog=catalog,
    ):
        inputs = torch.from_numpy(np.array(images, dtype=np.float32, copy=True))
        inputs.mul_(1.0 / 255.0).sub_(mean).div_(std)
        inputs = inputs.unsqueeze(1).to(device, non_blocking=True)
        probability = ensemble_up_probabilities(models, inputs)
        probabilities.append(probability.cpu().numpy())
        emitted_rows.append(source_rows)
        processed += len(source_rows)
        if progress_every > 0 and processed // progress_every != (
            processed - len(source_rows)
        ) // progress_every:
            print(f"{year}: scored {processed:,}/{len(selected_rows):,}", flush=True)
        del inputs, probability

    observed_rows = np.concatenate(emitted_rows) if emitted_rows else np.array([], dtype=np.int64)
    if not np.array_equal(observed_rows, selected_rows):
        raise AssertionError(f"Streamed image rows are misaligned for {year}")
    selected["cnn_prob_up"] = (
        np.concatenate(probabilities) if probabilities else np.array([], dtype=float)
    )
    if selected["cnn_prob_up"].isna().any() or not selected["cnn_prob_up"].between(0, 1).all():
        raise ValueError(f"Invalid CNN probabilities for {year}")
    selected["archive_year"] = year
    selected = selected.reset_index(drop=True)
    audit = {
        "year": year,
        "official_rows": int(profile["label_rows"]),
        "professor_universe_rows": int(len(formation)),
        "matched_and_scored_rows": int(len(selected)),
        "coverage_rate": float(len(selected) / len(formation)) if len(formation) else math.nan,
        "score_min": float(selected["cnn_prob_up"].min()),
        "score_mean": float(selected["cnn_prob_up"].mean()),
        "score_max": float(selected["cnn_prob_up"].max()),
        "seconds": time.perf_counter() - started,
    }
    return selected, audit


def build_monthly_portfolios(
    assignments: pd.DataFrame, *, weight_column: str | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usable = assignments.dropna(
        subset=["holding_month", "cnn_decile", "next_month_return"]
    ).copy()
    usable["portfolio"] = "Q" + usable["cnn_decile"].astype(int).astype(str)
    if weight_column is None:
        monthly = (
            usable.groupby(["holding_month", "portfolio"], as_index=False, observed=True)
            .agg(return_value=("next_month_return", "mean"), constituents=("PERMNO", "size"))
            .rename(columns={"return_value": "return"})
        )
        weighting = "equal_weight"
    else:
        usable[weight_column] = pd.to_numeric(usable[weight_column], errors="coerce")
        usable = usable.loc[usable[weight_column].gt(0)].copy()
        usable["weighted_return"] = usable["next_month_return"] * usable[weight_column]
        monthly = usable.groupby(
            ["holding_month", "portfolio"], as_index=False, observed=True
        ).agg(
            weighted_return=("weighted_return", "sum"),
            total_weight=(weight_column, "sum"),
            constituents=("PERMNO", "size"),
        )
        monthly["return"] = monthly["weighted_return"] / monthly["total_weight"]
        monthly = monthly[["holding_month", "portfolio", "return", "constituents"]]
        weighting = "value_weight"
    expected = {f"Q{i}" for i in range(1, 11)}
    observed = set(monthly["portfolio"].unique())
    if missing := sorted(expected.difference(observed)):
        raise ValueError(f"Missing CNN deciles: {missing}")
    with_spread = add_long_short(
        monthly[["holding_month", "portfolio", "return"]],
        long_label="Q10",
        short_label="Q1",
        spread_label="Q10-Q1",
    )
    audit = {
        "weighting": weighting,
        "usable_assignment_rows": int(len(usable)),
        "missing_next_month_return": int(assignments["next_month_return"].isna().sum()),
        "return_months": int(with_spread["holding_month"].nunique()),
        "return_month_start": str(with_spread["holding_month"].min()),
        "return_month_end": str(with_spread["holding_month"].max()),
        "average_constituents": {
            str(key): float(value)
            for key, value in monthly.groupby("portfolio", observed=True)["constituents"].mean().items()
        },
    }
    return with_spread.sort_values(["holding_month", "portfolio"]), audit


def _figure_data(equal_summary: pd.DataFrame, value_summary: pd.DataFrame) -> pd.DataFrame:
    def select(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        result = frame.loc[frame["portfolio"].str.fullmatch(r"Q\d+")].copy()
        result["decile"] = result["portfolio"].str[1:].astype(int)
        result = result.sort_values("decile")
        if result["decile"].tolist() != list(range(1, 11)):
            raise ValueError(f"Expected deciles 1 through 10 for {prefix}")
        return result[
            ["decile", "annualized_mean_return", "annualized_volatility"]
        ].rename(
            columns={
                "annualized_mean_return": f"{prefix}_annualized_mean_return",
                "annualized_volatility": f"{prefix}_annualized_volatility",
            }
        )

    return select(equal_summary, "equal_weight").merge(
        select(value_summary, "value_weight"),
        on="decile",
        how="inner",
        validate="one_to_one",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.start_year > args.end_year:
        raise ValueError("start-year cannot exceed end-year")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full OOS ensemble scoring run")

    image_zip = args.image_zip.expanduser().resolve()
    professor_csv = args.professor_csv.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    archive_hash, hash_matches = verify_official_archive_hash(
        image_zip, skip=args.skip_sha256
    )
    if args.skip_sha256:
        archive_hash = sha256_file(image_zip)
        hash_matches = archive_hash == EXPECTED_ARCHIVE_SHA256
    summary = _load_summary(checkpoint_dir)
    if archive_hash != summary["archive"]["sha256"]:
        raise ValueError("OOS image archive differs from the training archive")
    normalization = summary["normalization"]

    professor, professor_audit = load_professor_prices(professor_csv)
    formation_start = f"{args.start_year:04d}-01"
    formation_end = f"{args.end_year:04d}-12"
    formation = official_formation_universe(
        professor, formation_start=formation_start, formation_end=formation_end
    )
    catalog = catalog_archive(image_zip)
    available_years = {item.year for item in catalog}
    requested_years = list(range(args.start_year, args.end_year + 1))
    missing_years = sorted(set(requested_years).difference(available_years))
    if missing_years:
        raise ValueError(f"Image archive lacks requested years: {missing_years}")

    device = torch.device("cuda:0")
    models, model_audit = _load_models(checkpoint_dir, summary, device)
    score_frames: list[pd.DataFrame] = []
    year_audits: list[dict[str, Any]] = []
    for year in requested_years:
        year_formation = formation.loc[
            formation["formation_month"].str.startswith(f"{year:04d}-")
        ].copy()
        scored, year_audit = score_year(
            image_zip=image_zip,
            year=year,
            catalog=catalog,
            formation=year_formation,
            models=models,
            device=device,
            mean=float(normalization["mean"]),
            std=float(normalization["std"]),
            batch_size=args.batch_size,
            progress_every=args.progress_every,
        )
        score_frames.append(scored)
        year_audits.append(year_audit)
        print(
            f"{year}: complete, {len(scored):,} matched rows in {year_audit['seconds']:.1f}s",
            flush=True,
        )

    scores = pd.concat(score_frames, ignore_index=True)
    if scores.duplicated(["PERMNO", "formation_month"]).any():
        raise ValueError("CNN scores contain duplicate stock-month keys")
    if not scores["cnn_prob_up"].between(0, 1).all():
        raise ValueError("CNN scores fall outside [0, 1]")

    signal = formation[
        ["PERMNO", "date", "formation_month", "EXCHCD", "TICKER", "COMNAM"]
    ].rename(columns={"date": "formation_date"}).merge(
        scores,
        on=["PERMNO", "formation_month"],
        how="left",
        validate="one_to_one",
    )
    exact_date = signal["formation_date"].eq(signal["official_date"])
    score_coverage = signal["cnn_prob_up"].notna()
    signal = signal.loc[score_coverage].copy()
    signal["holding_month"] = (
        pd.PeriodIndex(signal["formation_month"], freq="M") + 1
    ).astype(str)
    signal["cnn_decile"] = assign_cnn_deciles(signal).astype("int64")
    signal["cnn_rank_pct"] = signal.groupby(
        "formation_month", sort=True, observed=True
    )["cnn_prob_up"].transform(month_rank_percentile)
    signal["cnn_score_signed"] = 2.0 * signal["cnn_rank_pct"] - 1.0
    signal = signal.sort_values(["formation_month", "PERMNO"]).reset_index(drop=True)

    month_sizes = signal.groupby(
        ["formation_month", "cnn_decile"], observed=True
    ).size().unstack("cnn_decile")
    if sorted(month_sizes.columns.tolist()) != list(range(1, 11)):
        raise ValueError("One or more months do not contain all ten CNN deciles")
    if not (month_sizes.max(axis=1) - month_sizes.min(axis=1)).le(1).all():
        raise AssertionError("CNN decile sizes differ by more than one within a month")

    outcomes, outcome_audit = build_adj_price_outcomes(
        professor, formation_start=formation_start, formation_end=formation_end
    )
    assignments = signal.merge(
        outcomes[
            [
                "PERMNO",
                "formation_month",
                "holding_month",
                "formation_adj_prc",
                "holding_adj_prc",
                "next_month_return",
            ]
        ],
        on=["PERMNO", "formation_month", "holding_month"],
        how="left",
        validate="one_to_one",
    )

    equal_monthly, equal_audit = build_monthly_portfolios(assignments)
    value_monthly, value_audit = build_monthly_portfolios(
        assignments, weight_column="formation_market_cap"
    )
    equal_summary = summarize_portfolios(equal_monthly)
    value_summary = summarize_portfolios(value_monthly)
    figure_data = _figure_data(equal_summary, value_summary)

    signal_path = output_dir / "cnn_signal_scores_2001_2019.csv.gz"
    assignment_path = output_dir / "cnn_assignments_with_outcomes.csv.gz"
    signal.to_csv(signal_path, index=False, compression="gzip")
    assignments.to_csv(assignment_path, index=False, compression="gzip")
    equal_monthly.to_csv(output_dir / "cnn_monthly_returns_equal.csv", index=False)
    value_monthly.to_csv(output_dir / "cnn_monthly_returns_value.csv", index=False)
    equal_summary.to_csv(output_dir / "cnn_summary_equal.csv", index=False)
    value_summary.to_csv(output_dir / "cnn_summary_value.csv", index=False)
    figure_data.to_csv(output_dir / "cnn_decile_figure_data.csv", index=False)

    audit = {
        "status": "complete",
        "archive": {
            "basename": image_zip.name,
            "sha256": archive_hash,
            "sha256_matches_expected": hash_matches,
        },
        "training_summary": str(checkpoint_dir / "summary.json"),
        "models": model_audit,
        "normalization": normalization,
        "professor": professor_audit,
        "formation_period": [formation_start, formation_end],
        "holding_period": [
            str(assignments["holding_month"].min()),
            str(assignments["holding_month"].max()),
        ],
        "universe": "Professor CSV formation-month rows with EXCHCD in {1,2,3}",
        "score_definition": "Mean across five frozen models of P(Ret_month > 0)",
        "decile_definition": "Within each formation month: Q1 lowest CNN probability, Q10 highest; ties by PERMNO",
        "timing_rule": "Formation-month t image and score; exact adjacent-month ADJ_PRC return at t+1",
        "long_short_direction": "Q10 high CNN probability minus Q1 low CNN probability",
        "formation_rows": int(len(formation)),
        "scored_rows": int(len(signal)),
        "score_coverage_rate": float(len(signal) / len(formation)),
        "exact_date_matches_among_scored": int(exact_date.loc[score_coverage].sum()),
        "exact_date_match_rate_among_scored": float(exact_date.loc[score_coverage].mean()),
        "duplicate_score_keys": int(signal.duplicated(["PERMNO", "formation_month"]).sum()),
        "year_scoring": year_audits,
        "outcomes": outcome_audit,
        "equal_weight": equal_audit,
        "value_weight": value_audit,
        "value_weight_source": "Absolute official label MarketCap at formation month",
        "annualization": {
            "mean": "12 * monthly arithmetic mean",
            "volatility": "sqrt(12) * monthly sample standard deviation",
            "newey_west_lag": 4,
        },
        "outputs": {
            "signal": signal_path.name,
            "assignments": assignment_path.name,
            "figure_data": "cnn_decile_figure_data.csv",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(audit, output_dir / "cnn_oos_deciles_audit.json")
    print(f"Wrote CNN OOS decile analysis to {output_dir}", flush=True)
    return audit


def main() -> None:
    args = parser().parse_args()
    try:
        run(args)
    except BaseException as error:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        temporary = output_dir / "cnn_oos_deciles_failure.json.partial"
        temporary.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir / "cnn_oos_deciles_failure.json")
        raise


if __name__ == "__main__":
    main()
