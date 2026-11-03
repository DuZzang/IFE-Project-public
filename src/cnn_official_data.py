from __future__ import annotations

import hashlib
import io
import math
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


IMAGE_SHAPE = (64, 60)
IMAGE_BYTES = int(np.prod(IMAGE_SHAPE))
EXPECTED_ARCHIVE_SHA256 = (
    "BE0E9BB378D705C3C842BDADCD6A9961A0A768603E9EDA9C099228A07F360DCC"
)
REQUIRED_LABEL_COLUMNS = (
    "Date",
    "StockID",
    "MarketCap",
    "Ret_5d",
    "Ret_20d",
    "Ret_60d",
    "Ret_month",
    "EWMA_vol",
)

_MEMBER_PATTERN = re.compile(
    r"^monthly_20d/20d_month_has_vb_\[20\]_ma_(?P<year>\d{4})_"
    r"(?P<kind>images\.dat|labels_w_delay\.feather)$"
)


@dataclass(frozen=True)
class ArchivePartition:
    year: int
    image_member: str
    label_member: str
    image_uncompressed_bytes: int
    image_compressed_bytes: int
    label_uncompressed_bytes: int
    label_compressed_bytes: int

    @property
    def image_count(self) -> int:
        return self.image_uncompressed_bytes // IMAGE_BYTES

    def to_dict(self) -> dict[str, int | str]:
        payload = asdict(self)
        payload["image_count"] = self.image_count
        return payload


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_official_archive_hash(
    path: str | Path,
    *,
    skip: bool = False,
) -> tuple[str | None, bool | None]:
    """Verify the known official archive, with an explicit opt-out for research copies."""
    if skip:
        return None, None
    observed = sha256_file(path)
    if observed != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(
            "Official image archive SHA-256 mismatch. Refusing to continue. "
            "Use --skip-sha256 only after independently verifying the archive."
        )
    return observed, True


def catalog_archive(zip_path: str | Path) -> list[ArchivePartition]:
    """Return validated annual image/label pairs from the official archive."""
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Image archive does not exist: {zip_path}")

    members: dict[int, dict[str, zipfile.ZipInfo]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/"):
                continue
            match = _MEMBER_PATTERN.fullmatch(info.filename)
            if match is None:
                continue
            year = int(match.group("year"))
            kind = "image" if match.group("kind") == "images.dat" else "label"
            partition = members.setdefault(year, {})
            if kind in partition:
                raise ValueError(f"Duplicate {kind} member for {year}: {info.filename}")
            partition[kind] = info

    if not members:
        raise ValueError(f"No monthly_20d partitions found in {zip_path}")

    result: list[ArchivePartition] = []
    for year, pair in sorted(members.items()):
        missing = {"image", "label"}.difference(pair)
        if missing:
            raise ValueError(f"Incomplete archive partition for {year}: missing {sorted(missing)}")
        image_info = pair["image"]
        label_info = pair["label"]
        remainder = image_info.file_size % IMAGE_BYTES
        if remainder:
            raise ValueError(
                f"Image bytes for {year} are not divisible by {IMAGE_BYTES}: "
                f"{image_info.file_size} bytes ({remainder} trailing bytes)"
            )
        result.append(
            ArchivePartition(
                year=year,
                image_member=image_info.filename,
                label_member=label_info.filename,
                image_uncompressed_bytes=int(image_info.file_size),
                image_compressed_bytes=int(image_info.compress_size),
                label_uncompressed_bytes=int(label_info.file_size),
                label_compressed_bytes=int(label_info.compress_size),
            )
        )
    return result


def _partition_for_year(
    zip_path: str | Path,
    year: int,
    catalog: Sequence[ArchivePartition] | None = None,
) -> ArchivePartition:
    matches = [item for item in (catalog or catalog_archive(zip_path)) if item.year == year]
    if len(matches) != 1:
        available = [item.year for item in (catalog or catalog_archive(zip_path))]
        raise KeyError(f"Year {year} is unavailable; archive years are {available}")
    return matches[0]


def read_year_labels(
    zip_path: str | Path,
    year: int,
    *,
    columns: Sequence[str] | None = None,
    catalog: Sequence[ArchivePartition] | None = None,
) -> pd.DataFrame:
    """Read one small annual Feather label member while preserving row order."""
    partition = _partition_for_year(zip_path, year, catalog)
    with zipfile.ZipFile(zip_path) as archive:
        payload = archive.read(partition.label_member)
    frame = pd.read_feather(io.BytesIO(payload), columns=columns)
    if columns is None:
        missing = sorted(set(REQUIRED_LABEL_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"Label partition {year} is missing columns: {missing}")
    return frame


def _read_exact(source: zipfile.ZipExtFile, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        block = source.read(remaining)
        if not block:
            break
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def iter_year_image_batches(
    zip_path: str | Path,
    year: int,
    *,
    batch_size: int = 256,
    max_images: int | None = None,
    catalog: Sequence[ArchivePartition] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream `(offset, uint8[batch, 64, 60])` without extracting the `.dat`."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_images is not None and max_images < 0:
        raise ValueError("max_images cannot be negative")

    partition = _partition_for_year(zip_path, year, catalog)
    requested_images = partition.image_count
    if max_images is not None:
        requested_images = min(requested_images, max_images)

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(partition.image_member) as source:
            offset = 0
            while offset < requested_images:
                n_images = min(batch_size, requested_images - offset)
                expected_bytes = n_images * IMAGE_BYTES
                payload = _read_exact(source, expected_bytes)
                if len(payload) != expected_bytes:
                    raise EOFError(
                        f"Image member ended early at image {offset}: expected "
                        f"{expected_bytes} bytes, read {len(payload)}"
                    )
                batch = np.frombuffer(payload, dtype=np.uint8).reshape(
                    n_images, *IMAGE_SHAPE
                )
                batch.setflags(write=False)
                yield offset, batch
                offset += n_images

            if max_images is None and source.read(1):
                raise ValueError(f"Image member for {year} has trailing bytes")


def iter_filtered_year_image_batches(
    zip_path: str | Path,
    year: int,
    keep_mask: Sequence[bool],
    *,
    batch_size: int = 256,
    catalog: Sequence[ArchivePartition] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield source row indices and images selected by an image-aligned mask."""
    partition = _partition_for_year(zip_path, year, catalog)
    mask = np.asarray(keep_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != partition.image_count:
        raise ValueError(
            f"Mask length for {year} must equal {partition.image_count}; got {mask.shape}"
        )
    for offset, batch in iter_year_image_batches(
        zip_path, year, batch_size=batch_size, catalog=catalog
    ):
        local_mask = mask[offset : offset + len(batch)]
        if not local_mask.any():
            continue
        source_rows = np.flatnonzero(local_mask).astype(np.int64) + offset
        selected = batch[local_mask]
        selected.setflags(write=False)
        yield source_rows, selected


def normalize_label_frame(
    frame: pd.DataFrame,
    *,
    archive_year: int,
) -> pd.DataFrame:
    """Normalize official label keys but never sort the image-aligned rows."""
    missing = sorted(set(REQUIRED_LABEL_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Official labels are missing columns: {missing}")
    normalized = frame.copy()
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    normalized["PERMNO"] = pd.to_numeric(
        normalized["StockID"], errors="coerce"
    ).astype("Int64")
    normalized["month"] = normalized["Date"].dt.strftime("%Y-%m")
    normalized["archive_year"] = int(archive_year)
    return normalized


def profile_label_partition(
    partition: ArchivePartition,
    labels: pd.DataFrame,
) -> tuple[dict[str, int | float | str | bool | None], pd.DataFrame]:
    """Profile one annual label file and return normalized rows for matching."""
    normalized = normalize_label_frame(labels, archive_year=partition.year)
    valid_keys = normalized.dropna(subset=["Date", "PERMNO", "month"])
    duplicated = valid_keys.duplicated(["PERMNO", "month"], keep=False)
    duplicate_excess = len(valid_keys) - len(
        valid_keys.drop_duplicates(["PERMNO", "month"])
    )
    market_cap = pd.to_numeric(normalized["MarketCap"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    ret_20d = pd.to_numeric(normalized["Ret_20d"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    ret_month = pd.to_numeric(normalized["Ret_month"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )

    observed_years = normalized["Date"].dt.year
    profile: dict[str, int | float | str | bool | None] = {
        "year": partition.year,
        "label_rows": int(len(normalized)),
        "image_count": int(partition.image_count),
        "image_label_delta": int(partition.image_count - len(normalized)),
        "image_label_count_match": bool(partition.image_count == len(normalized)),
        "image_uncompressed_bytes": int(partition.image_uncompressed_bytes),
        "image_compressed_bytes": int(partition.image_compressed_bytes),
        "label_uncompressed_bytes": int(partition.label_uncompressed_bytes),
        "label_compressed_bytes": int(partition.label_compressed_bytes),
        "min_date": (
            normalized["Date"].min().date().isoformat()
            if normalized["Date"].notna().any()
            else None
        ),
        "max_date": (
            normalized["Date"].max().date().isoformat()
            if normalized["Date"].notna().any()
            else None
        ),
        "unique_months": int(normalized["month"].nunique(dropna=True)),
        "unique_permnos": int(normalized["PERMNO"].nunique(dropna=True)),
        "null_dates": int(normalized["Date"].isna().sum()),
        "invalid_stock_ids": int(normalized["PERMNO"].isna().sum()),
        "duplicate_key_rows": int(duplicated.sum()),
        "duplicate_key_excess": int(duplicate_excess),
        "date_year_mismatch_rows": int(
            ((observed_years.notna()) & observed_years.ne(partition.year)).sum()
        ),
        "missing_ret_20d": int((~np.isfinite(ret_20d)).sum()),
        "missing_ret_20d_rate": float((~np.isfinite(ret_20d)).mean()),
        "missing_ret_month": int((~np.isfinite(ret_month)).sum()),
        "missing_ret_month_rate": float((~np.isfinite(ret_month)).mean()),
        "missing_market_cap": int((~np.isfinite(market_cap)).sum()),
        "negative_market_cap": int(
            (np.isfinite(market_cap) & (market_cap < 0)).sum()
        ),
        "zero_market_cap": int(
            (np.isfinite(market_cap) & (market_cap == 0)).sum()
        ),
    }
    return profile, normalized


def prepare_classification_labels(
    labels: pd.DataFrame,
    *,
    target_column: str = "Ret_month",
    input_date_cutoff_exclusive: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int | str | None]]:
    """Create finite monthly labels and an image-aligned keep mask.

    The official code maps missing returns to class zero before filtering. This
    adapter excludes them first, requires finite positive lagged EWMA variance,
    preserves source-row positions, and converts signed CRSP market cap to its
    absolute value for portfolio weighting.
    """
    allowed_targets = {"Ret_5d", "Ret_20d", "Ret_60d", "Ret_month"}
    if target_column not in allowed_targets:
        raise ValueError(
            f"target_column must be one of {sorted(allowed_targets)}; got {target_column}"
        )
    required = {target_column, "EWMA_vol", "MarketCap"}
    if input_date_cutoff_exclusive is not None:
        required.add("Date")
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"Cannot prepare labels; missing columns: {missing}")

    target = pd.to_numeric(labels[target_column], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    ewma_vol = pd.to_numeric(labels["EWMA_vol"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    market_cap = pd.to_numeric(labels["MarketCap"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    finite_target = np.isfinite(target)
    valid_volatility = np.isfinite(ewma_vol) & (ewma_vol > 0)
    if input_date_cutoff_exclusive is None:
        time_eligible = np.ones(len(labels), dtype=bool)
        cutoff_value = None
    else:
        cutoff = pd.Timestamp(input_date_cutoff_exclusive)
        dates = pd.to_datetime(labels["Date"], errors="coerce")
        time_eligible = dates.notna().to_numpy() & dates.lt(cutoff).to_numpy()
        cutoff_value = cutoff.date().isoformat()
    keep_mask = finite_target & valid_volatility & time_eligible

    prepared = labels.loc[keep_mask].copy()
    prepared["source_row"] = np.flatnonzero(keep_mask).astype(np.int64)
    prepared["MarketCap"] = np.abs(market_cap[keep_mask])
    prepared["label"] = target[keep_mask].astype(float) > 0
    prepared["label"] = prepared["label"].astype(np.int8)
    audit: dict[str, int | str | None] = {
        "target_column": target_column,
        "input_date_cutoff_exclusive": cutoff_value,
        "input_rows": int(len(labels)),
        "kept_rows": int(keep_mask.sum()),
        "dropped_total": int((~keep_mask).sum()),
        "rows_with_missing_or_nonfinite_target": int((~finite_target).sum()),
        "rows_with_missing_nonfinite_or_nonpositive_ewma_vol": int(
            (~valid_volatility).sum()
        ),
        "dropped_target_only": int((~finite_target & valid_volatility).sum()),
        "dropped_ewma_only": int((finite_target & ~valid_volatility).sum()),
        "dropped_target_and_ewma": int((~finite_target & ~valid_volatility).sum()),
        "dropped_at_or_after_cutoff_or_invalid_date": int((~time_eligible).sum()),
        "invalid_market_cap_rows_retained_for_training_only": int(
            (~np.isfinite(market_cap) & keep_mask).sum()
        ),
        "negative_market_cap_rows_converted_to_absolute": int(
            (np.isfinite(market_cap) & (market_cap < 0) & keep_mask).sum()
        ),
    }
    return prepared, keep_mask, audit


def validate_label_partition(
    partition: ArchivePartition,
    labels: pd.DataFrame,
) -> tuple[dict[str, int | float | str | bool | None], pd.DataFrame]:
    """Reject structural defects that would break image-to-label alignment."""
    profile, normalized = profile_label_partition(partition, labels)
    failures: list[str] = []
    if not profile["image_label_count_match"]:
        failures.append(
            f"image_count={profile['image_count']} label_rows={profile['label_rows']}"
        )
    for field in (
        "null_dates",
        "invalid_stock_ids",
        "duplicate_key_excess",
        "date_year_mismatch_rows",
    ):
        if int(profile[field]):
            failures.append(f"{field}={profile[field]}")
    if failures:
        raise ValueError(
            f"Invalid official label partition {partition.year}: " + ", ".join(failures)
        )
    return profile, normalized


def read_professor_monthly(
    csv_path: str | Path,
    *,
    start_year: int,
    end_year: int,
    chunksize: int = 250_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read only key fields and retain the requested calendar-year window."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Professor CSV does not exist: {csv_path}")
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = sorted({"PERMNO", "date"}.difference(header))
    if missing:
        raise ValueError(f"Professor CSV is missing columns: {missing}")
    usecols = [column for column in ("PERMNO", "date", "EXCHCD") if column in header]

    chunks: list[pd.DataFrame] = []
    total_rows = 0
    invalid_dates = 0
    invalid_permnos = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        total_rows += len(chunk)
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["PERMNO"] = pd.to_numeric(chunk["PERMNO"], errors="coerce").astype(
            "Int64"
        )
        invalid_dates += int(chunk["date"].isna().sum())
        invalid_permnos += int(chunk["PERMNO"].isna().sum())
        chunks.append(chunk)

    if chunks:
        source_frame = pd.concat(chunks, ignore_index=True)
    else:
        source_frame = pd.DataFrame(columns=usecols)
    source_frame["month"] = source_frame["date"].dt.strftime("%Y-%m")
    valid_source = source_frame.dropna(subset=["PERMNO", "date", "month"]).copy()
    valid_source = valid_source.sort_values(
        ["PERMNO", "date"], kind="mergesort"
    )
    valid_source["source_observation_number"] = (
        valid_source.groupby("PERMNO", sort=False).cumcount() + 1
    )
    valid_source["source_observation_count"] = valid_source.groupby(
        "PERMNO", sort=False
    )["PERMNO"].transform("size")
    rank_columns = valid_source[
        ["source_observation_number", "source_observation_count"]
    ].reindex(source_frame.index)
    source_frame[["source_observation_number", "source_observation_count"]] = rank_columns
    mask = source_frame["date"].dt.year.between(
        start_year, end_year, inclusive="both"
    )
    frame = source_frame.loc[mask].copy()
    frame["month"] = frame["date"].dt.strftime("%Y-%m")
    if "EXCHCD" in frame:
        frame["EXCHCD"] = pd.to_numeric(frame["EXCHCD"], errors="coerce")
        frame["exchange_eligible"] = frame["EXCHCD"].isin([1, 2, 3])
    else:
        frame["EXCHCD"] = np.nan
        frame["exchange_eligible"] = True
    frame["is_first_source_observation"] = frame["source_observation_number"].eq(1)
    frame["within_first_3_source_observations"] = frame[
        "source_observation_number"
    ].le(3)
    frame["is_last_source_observation"] = frame["source_observation_number"].eq(
        frame["source_observation_count"]
    )

    valid = frame.dropna(subset=["PERMNO", "date", "month"])
    month_duplicate_rows = valid.duplicated(["PERMNO", "month"], keep=False)
    exact_duplicate_rows = valid.duplicated(["PERMNO", "date"], keep=False)
    summary: dict[str, object] = {
        "source_rows": int(total_rows),
        "source_valid_rows": int(len(valid_source)),
        "source_min_date": (
            valid_source["date"].min().date().isoformat()
            if not valid_source.empty
            else None
        ),
        "source_max_date": (
            valid_source["date"].max().date().isoformat()
            if not valid_source.empty
            else None
        ),
        "source_rows_before_window": int(
            (valid_source["date"].dt.year < start_year).sum()
        ),
        "source_rows_after_window": int(
            (valid_source["date"].dt.year > end_year).sum()
        ),
        "source_rows_by_year": {
            str(int(year)): int(count)
            for year, count in valid_source["date"].dt.year.value_counts().sort_index().items()
        },
        "window_rows": int(len(frame)),
        "valid_window_rows": int(len(valid)),
        "source_invalid_dates": int(invalid_dates),
        "source_invalid_permnos": int(invalid_permnos),
        "window_min_date": (
            valid["date"].min().date().isoformat() if not valid.empty else None
        ),
        "window_max_date": (
            valid["date"].max().date().isoformat() if not valid.empty else None
        ),
        "window_unique_months": int(valid["month"].nunique()),
        "window_unique_permnos": int(valid["PERMNO"].nunique()),
        "window_month_key_duplicate_rows": int(month_duplicate_rows.sum()),
        "window_month_key_duplicate_excess": int(
            len(valid) - len(valid.drop_duplicates(["PERMNO", "month"]))
        ),
        "window_exact_key_duplicate_rows": int(exact_duplicate_rows.sum()),
        "window_exact_key_duplicate_excess": int(
            len(valid) - len(valid.drop_duplicates(["PERMNO", "date"]))
        ),
        "window_exchange_ineligible_rows": int((~valid["exchange_eligible"]).sum()),
        "window_missing_exchcd": int(valid["EXCHCD"].isna().sum()),
    }
    return frame, summary


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _finite_numeric_mask(series: pd.Series) -> np.ndarray:
    return np.isfinite(
        pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    )


def _coverage_for_scope(
    official: pd.DataFrame,
    professor: pd.DataFrame,
    *,
    scope: str,
) -> tuple[dict[str, int | float | str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    professor_scope = professor.loc[
        professor["exchange_eligible"] if scope == "exchange_eligible" else slice(None)
    ].copy()
    professor_keys = (
        professor_scope.dropna(subset=["PERMNO", "date", "month"])
        .sort_values(["PERMNO", "date"], kind="mergesort")
        .drop_duplicates(["PERMNO", "month"], keep="last")
        .rename(columns={"date": "professor_date"})
    )
    official_keys = (
        official.dropna(subset=["PERMNO", "Date", "month"])
        .sort_values(["PERMNO", "Date"], kind="mergesort")
        .drop_duplicates(["PERMNO", "month"], keep="last")
        .rename(columns={"Date": "official_date"})
    )

    professor_columns = [
        "PERMNO",
        "month",
        "professor_date",
        "EXCHCD",
        "source_observation_number",
        "source_observation_count",
        "is_first_source_observation",
        "within_first_3_source_observations",
        "is_last_source_observation",
    ]
    official_columns = [
        "PERMNO",
        "month",
        "official_date",
        "Ret_20d",
        "Ret_month",
        "EWMA_vol",
        "MarketCap",
    ]
    matched = professor_keys[professor_columns].merge(
        official_keys[official_columns],
        on=["PERMNO", "month"],
        how="inner",
        validate="one_to_one",
    )
    matched["exact_date_match"] = matched["professor_date"].eq(
        matched["official_date"]
    )
    matched_target = pd.to_numeric(matched["Ret_month"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    matched_volatility = pd.to_numeric(
        matched["EWMA_vol"], errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)
    matched["model_label_eligible"] = np.isfinite(matched_target) & np.isfinite(
        matched_volatility
    ) & (matched_volatility > 0)
    matched["year"] = matched["month"].str.slice(0, 4).astype(int)

    professor_only = professor_keys[professor_columns].merge(
        official_keys[["PERMNO", "month"]],
        on=["PERMNO", "month"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    professor_only = professor_only.loc[professor_only["_merge"].eq("left_only")].drop(
        columns="_merge"
    )
    official_only = official_keys[
        ["PERMNO", "month", "official_date", "Ret_20d", "Ret_month"]
    ].merge(
        professor_keys[["PERMNO", "month"]],
        on=["PERMNO", "month"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    official_only = official_only.loc[official_only["_merge"].eq("left_only")].drop(
        columns="_merge"
    )

    ret20_all = pd.to_numeric(matched["Ret_20d"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    retmonth_all = matched_target
    finite_ret20 = np.isfinite(ret20_all)
    finite_retmonth = np.isfinite(retmonth_all)
    both_returns = finite_ret20 & finite_retmonth
    ret20 = ret20_all[both_returns]
    retmonth = retmonth_all[both_returns]
    equal_returns = np.isclose(ret20, retmonth, rtol=1e-10, atol=1e-12)
    sign_disagreement = (ret20 > 0) != (retmonth > 0)

    result: dict[str, int | float | str] = {
        "scope": scope,
        "professor_keys": int(len(professor_keys)),
        "official_keys": int(len(official_keys)),
        "matched_keys": int(len(matched)),
        "professor_only_keys": int(len(professor_only)),
        "official_only_keys": int(len(official_only)),
        "professor_only_first_source_observation": int(
            professor_only["is_first_source_observation"].sum()
        ),
        "professor_only_within_first_3_source_observations": int(
            professor_only["within_first_3_source_observations"].sum()
        ),
        "professor_only_last_source_observation": int(
            professor_only["is_last_source_observation"].sum()
        ),
        "professor_coverage_rate": _safe_rate(len(matched), len(professor_keys)),
        "official_coverage_rate": _safe_rate(len(matched), len(official_keys)),
        "exact_date_matches": int(matched["exact_date_match"].sum()),
        "exact_date_match_rate_among_matches": _safe_rate(
            int(matched["exact_date_match"].sum()), len(matched)
        ),
        "matched_ret_20d_available": int(finite_ret20.sum()),
        "matched_ret_20d_available_rate": float(finite_ret20.mean())
        if len(matched)
        else math.nan,
        "matched_ret_month_available": int(finite_retmonth.sum()),
        "matched_ret_month_available_rate": float(finite_retmonth.mean())
        if len(matched)
        else math.nan,
        "matched_model_label_eligible": int(matched["model_label_eligible"].sum()),
        "matched_model_label_eligible_rate": float(
            matched["model_label_eligible"].mean()
        )
        if len(matched)
        else math.nan,
        "ret_20d_month_comparable": int(len(ret20)),
        "ret_20d_month_equal": int(equal_returns.sum()),
        "ret_20d_month_equal_rate": _safe_rate(int(equal_returns.sum()), len(ret20)),
        "ret_20d_month_sign_disagreements": int(sign_disagreement.sum()),
        "ret_20d_month_sign_disagreement_rate": _safe_rate(
            int(sign_disagreement.sum()), len(ret20)
        ),
        "ret_20d_month_mean_absolute_difference": float(
            np.mean(np.abs(ret20 - retmonth))
        )
        if len(ret20)
        else math.nan,
        "professor_unique_permnos": int(professor_keys["PERMNO"].nunique()),
        "official_unique_permnos": int(official_keys["PERMNO"].nunique()),
        "matched_unique_permnos": int(matched["PERMNO"].nunique()),
    }
    return result, matched, professor_only, official_only


def match_by_exchange(
    official: pd.DataFrame,
    professor: pd.DataFrame,
) -> pd.DataFrame:
    """Return professor-to-official coverage separately for each EXCHCD value."""
    rows: list[dict[str, int | float | str]] = []
    observed = professor["EXCHCD"].drop_duplicates().tolist()
    observed.sort(key=lambda value: (pd.isna(value), str(value)))
    for value in observed:
        if pd.isna(value):
            subset = professor.loc[professor["EXCHCD"].isna()].copy()
            label = "<MISSING>"
        else:
            subset = professor.loc[professor["EXCHCD"].eq(value)].copy()
            numeric = float(value)
            label = str(int(numeric)) if numeric.is_integer() else str(numeric)
        summary, _, _, _ = _coverage_for_scope(official, subset, scope="all")
        rows.append(
            {
                "EXCHCD": label,
                "professor_keys": summary["professor_keys"],
                "matched_keys": summary["matched_keys"],
                "professor_only_keys": summary["professor_only_keys"],
                "professor_coverage_rate": summary["professor_coverage_rate"],
                "exact_date_match_rate_among_matches": summary[
                    "exact_date_match_rate_among_matches"
                ],
            }
        )
    return pd.DataFrame(rows)


def _period_coverage(
    official: pd.DataFrame,
    professor: pd.DataFrame,
    matched: pd.DataFrame,
    *,
    scope: str,
    frequency: str,
) -> pd.DataFrame:
    period_column = "year" if frequency == "year" else "month"
    if period_column == "year":
        official_period = official["month"].str.slice(0, 4).astype("Int64")
        professor_period = professor["month"].str.slice(0, 4).astype("Int64")
    else:
        official_period = official["month"]
        professor_period = professor["month"]

    off = official.assign(_period=official_period).dropna(
        subset=["PERMNO", "month", "_period"]
    )
    prof = professor.assign(_period=professor_period).dropna(
        subset=["PERMNO", "month", "_period"]
    )
    if scope == "exchange_eligible":
        prof = prof.loc[prof["exchange_eligible"]]
    off = off.drop_duplicates(["PERMNO", "month"])
    prof = prof.drop_duplicates(["PERMNO", "month"])

    periods = sorted(set(off["_period"]).union(prof["_period"]))
    rows: list[dict[str, int | float | str]] = []
    for period in periods:
        off_part = off.loc[off["_period"].eq(period)]
        prof_part = prof.loc[prof["_period"].eq(period)]
        match_part = matched.loc[
            matched[period_column].eq(period)
            if period_column in matched
            else matched["month"].eq(period)
        ]
        row: dict[str, int | float | str] = {
            "scope": scope,
            period_column: int(period) if period_column == "year" else str(period),
            "professor_keys": int(len(prof_part)),
            "official_keys": int(len(off_part)),
            "matched_keys": int(len(match_part)),
            "professor_coverage_rate": _safe_rate(len(match_part), len(prof_part)),
            "official_coverage_rate": _safe_rate(len(match_part), len(off_part)),
            "exact_date_matches": int(match_part["exact_date_match"].sum()),
            "exact_date_match_rate_among_matches": _safe_rate(
                int(match_part["exact_date_match"].sum()), len(match_part)
            ),
            "matched_ret_20d_available": int(
                _finite_numeric_mask(match_part["Ret_20d"]).sum()
            ),
            "matched_ret_20d_available_rate": float(
                _finite_numeric_mask(match_part["Ret_20d"]).mean()
            )
            if len(match_part)
            else math.nan,
            "matched_model_label_eligible": int(
                match_part["model_label_eligible"].sum()
            ),
            "matched_model_label_eligible_rate": float(
                match_part["model_label_eligible"].mean()
            )
            if len(match_part)
            else math.nan,
            "professor_unique_permnos": int(prof_part["PERMNO"].nunique()),
            "official_unique_permnos": int(off_part["PERMNO"].nunique()),
            "matched_unique_permnos": int(match_part["PERMNO"].nunique()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def match_universes(
    official: pd.DataFrame,
    professor: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, int | float | str]],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[pd.DataFrame, pd.DataFrame]],
]:
    """Compare both universes at the unique PERMNO-calendar-month grain."""
    summaries: dict[str, dict[str, int | float | str]] = {}
    annual: list[pd.DataFrame] = []
    monthly: list[pd.DataFrame] = []
    unmatched: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for scope in ("all", "exchange_eligible"):
        summary, matched, professor_only, official_only = _coverage_for_scope(
            official, professor, scope=scope
        )
        summaries[scope] = summary
        annual.append(
            _period_coverage(
                official, professor, matched, scope=scope, frequency="year"
            )
        )
        monthly.append(
            _period_coverage(
                official, professor, matched, scope=scope, frequency="month"
            )
        )
        unmatched[scope] = (professor_only, official_only)
    return (
        summaries,
        pd.concat(annual, ignore_index=True),
        pd.concat(monthly, ignore_index=True),
        unmatched,
    )
