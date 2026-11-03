from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from cnn_official_data import (
    EXPECTED_ARCHIVE_SHA256,
    IMAGE_BYTES,
    IMAGE_SHAPE,
    ArchivePartition,
    catalog_archive,
    prepare_classification_labels,
    read_year_labels,
    sha256_file,
    validate_label_partition,
    verify_official_archive_hash,
)


EXPECTED_YEARS = tuple(range(1993, 2001))
EXPECTED_ELIGIBLE_ROWS = 778_507
EXPECTED_MODEL_PARAMETERS = 708_866
DEFAULT_BATCH_SIZE = 128
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_LEARNING_RATE = 1e-4


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(
        description=(
            "Run exactly one FP32 train+validation epoch for the official I20 CNN. "
            "The script never runs an ensemble or OOS portfolio evaluation."
        )
    )
    result.add_argument("--image-zip", type=Path, required=True)
    result.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "data" / "raw" / "cnn_i20_benchmark",
        help="Git-ignored cache for the eight extracted 1993-2000 image members.",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "cnn_epoch_benchmark" / "benchmark.json",
    )
    result.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "cnn_epoch_benchmark.md",
    )
    result.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    result.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    result.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    result.add_argument("--seed", type=int, default=DEFAULT_SEED)
    result.add_argument("--num-workers", type=int, default=1)
    result.add_argument("--warmup-batches", type=int, default=10)
    result.add_argument("--progress-every", type=int, default=250)
    result.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Smoke-test only. Omit for the full training pass.",
    )
    result.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help="Smoke-test only. Omit for the full validation pass.",
    )
    result.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Only for an independently verified research copy; official hash is required by default.",
    )
    return result


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _nvidia_smi_snapshot() -> dict[str, str] | None:
    fields = (
        "name,memory.total,memory.used,memory.free,temperature.gpu,"
        "power.draw,power.limit,clocks.current.graphics,driver_version"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    names = fields.split(",")
    return dict(zip(names, values, strict=False))


def _cuda_preflight() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. The benchmark refuses a CPU fallback because it would "
            "not estimate the planned GPU training time."
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    arch_list = list(torch.cuda.get_arch_list())
    capability_tag = f"sm_{capability[0]}{capability[1]}"
    architecture_tag_found = any(
        item.startswith(capability_tag) or item.startswith(f"compute_{capability[0]}{capability[1]}")
        for item in arch_list
    )

    x = torch.randn((32, 64), device=device, requires_grad=True)
    y = (x @ x.transpose(0, 1)).square().mean()
    y.backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(y).item():
        raise RuntimeError("CUDA preflight produced a non-finite result")
    convolution = nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(2, 1)).to(device)
    image = torch.randn((2, 1, *IMAGE_SHAPE), device=device, requires_grad=True)
    convolution_loss = convolution(image).square().mean()
    convolution_loss.backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(convolution_loss).item():
        raise RuntimeError("CUDA convolution preflight produced a non-finite result")
    del convolution, convolution_loss, image, x, y

    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "device_name": properties.name,
        "device_capability": list(capability),
        "device_total_memory_bytes": int(properties.total_memory),
        "compiled_arch_list": arch_list,
        "device_architecture_tag_found": architecture_tag_found,
    }


def _crc32_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> int:
    checksum = 0
    with path.open("rb") as source:
        while block := source.read(chunk_bytes):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF


def _extract_member_atomic(
    archive: zipfile.ZipFile,
    partition: ArchivePartition,
    destination: Path,
    expected_crc32: int,
) -> tuple[float, bool, int]:
    if destination.is_file() and destination.stat().st_size == partition.image_uncompressed_bytes:
        cached_crc32 = _crc32_file(destination)
        if cached_crc32 == expected_crc32:
            return 0.0, True, cached_crc32
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    started = time.perf_counter()
    with archive.open(partition.image_member) as source, temporary.open("wb") as sink:
        shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
    extracted_bytes = temporary.stat().st_size
    if extracted_bytes != partition.image_uncompressed_bytes:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Extracted byte count mismatch for {partition.year}: "
            f"{extracted_bytes} != {partition.image_uncompressed_bytes}"
        )
    os.replace(temporary, destination)
    return time.perf_counter() - started, False, expected_crc32


@dataclass(frozen=True)
class YearCache:
    year: int
    image_path: Path
    image_count: int
    source_rows: np.ndarray
    labels: np.ndarray
    mean: float
    std: float
    extraction_seconds: float
    cache_reused: bool
    image_crc32: int

    @property
    def eligible_count(self) -> int:
        return int(len(self.labels))


def _official_year_mean_std(image_path: Path, image_count: int) -> tuple[float, float]:
    images = np.memmap(
        image_path, dtype=np.uint8, mode="r", shape=(image_count, *IMAGE_SHAPE)
    )
    sample = images[: min(50_000, image_count)]
    mean = float(sample.mean(dtype=np.float64) / 255.0)
    std = float(sample.std(dtype=np.float64) / 255.0)
    del images
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid normalization statistics: mean={mean}, std={std}")
    return mean, std


def prepare_cache(
    image_zip: Path,
    cache_dir: Path,
    *,
    skip_sha256: bool,
) -> tuple[list[YearCache], dict[str, Any]]:
    image_zip = image_zip.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    if skip_sha256:
        archive_hash = sha256_file(image_zip)
        hash_matches = archive_hash == EXPECTED_ARCHIVE_SHA256
    else:
        archive_hash, hash_matches = verify_official_archive_hash(
            image_zip, skip=False
        )
    catalog = catalog_archive(image_zip)
    by_year = {partition.year: partition for partition in catalog}
    missing = sorted(set(EXPECTED_YEARS).difference(by_year))
    if missing:
        raise ValueError(f"Official archive lacks benchmark years: {missing}")

    caches: list[YearCache] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(image_zip) as archive:
        for year in EXPECTED_YEARS:
            partition = by_year[year]
            labels = read_year_labels(image_zip, year, catalog=catalog)
            _, normalized = validate_label_partition(partition, labels)
            cutoff = "2000-12-01" if year == 2000 else None
            prepared, _, _ = prepare_classification_labels(
                normalized,
                target_column="Ret_month",
                input_date_cutoff_exclusive=cutoff,
            )
            image_path = cache_dir / f"{year}_images.dat"
            expected_crc32 = int(archive.getinfo(partition.image_member).CRC)
            extraction_seconds, cache_reused, image_crc32 = _extract_member_atomic(
                archive, partition, image_path, expected_crc32
            )
            if image_path.stat().st_size % IMAGE_BYTES:
                raise ValueError(f"Cached image file has trailing bytes: {image_path}")
            mean, std = _official_year_mean_std(image_path, partition.image_count)
            source_rows = prepared["source_row"].to_numpy(dtype=np.int64, copy=True)
            target = prepared["label"].to_numpy(dtype=np.int64, copy=True)
            if len(source_rows) != len(target):
                raise AssertionError(f"Image/label index mismatch for {year}")
            if len(source_rows) and (
                int(source_rows.min()) < 0
                or int(source_rows.max()) >= partition.image_count
            ):
                raise ValueError(f"Out-of-range source row for {year}")
            caches.append(
                YearCache(
                    year=year,
                    image_path=image_path,
                    image_count=partition.image_count,
                    source_rows=source_rows,
                    labels=target,
                    mean=mean,
                    std=std,
                    extraction_seconds=extraction_seconds,
                    cache_reused=cache_reused,
                    image_crc32=image_crc32,
                )
            )

    total_eligible = sum(item.eligible_count for item in caches)
    if not skip_sha256 and total_eligible != EXPECTED_ELIGIBLE_ROWS:
        raise ValueError(
            f"Official benchmark eligibility changed: {total_eligible} != "
            f"{EXPECTED_ELIGIBLE_ROWS}"
        )
    manifest: dict[str, Any] = {
        "archive_basename": image_zip.name,
        "archive_sha256": archive_hash,
        "archive_sha256_matches_expected": hash_matches,
        "expected_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "years": list(EXPECTED_YEARS),
        "normalization": (
            "official-compatible scalar mean/std from each year's first 50,000 "
            "unfiltered images after division by 255"
        ),
        "total_eligible_rows": total_eligible,
        "total_cached_image_bytes": sum(
            item.image_path.stat().st_size for item in caches
        ),
        "total_extraction_seconds_this_run": sum(
            item.extraction_seconds for item in caches
        ),
        "partitions": [
            {
                "year": item.year,
                "image_file": item.image_path.name,
                "image_count": item.image_count,
                "eligible_count": item.eligible_count,
                "normalization_mean": item.mean,
                "normalization_std": item.std,
                "extraction_seconds_this_run": item.extraction_seconds,
                "cache_reused": item.cache_reused,
                "image_crc32": f"{item.image_crc32:08X}",
            }
            for item in caches
        ],
    }
    _atomic_json(manifest, cache_dir / "cache_manifest.json")
    return caches, manifest


class MultiYearImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, caches: Sequence[YearCache]) -> None:
        self.caches = list(caches)
        self.cumulative = np.cumsum(
            [item.eligible_count for item in self.caches], dtype=np.int64
        )
        self._maps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return int(self.cumulative[-1]) if len(self.cumulative) else 0

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_maps"] = {}
        return state

    def close(self) -> None:
        for image_map in self._maps.values():
            memory_map = getattr(image_map, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        self._maps.clear()

    def __del__(self) -> None:
        self.close()

    def _map_for(self, position: int) -> np.memmap:
        if position not in self._maps:
            cache = self.caches[position]
            self._maps[position] = np.memmap(
                cache.image_path,
                dtype=np.uint8,
                mode="r",
                shape=(cache.image_count, *IMAGE_SHAPE),
            )
        return self._maps[position]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        position = bisect.bisect_right(self.cumulative, index)
        previous = 0 if position == 0 else int(self.cumulative[position - 1])
        local_index = index - previous
        cache = self.caches[position]
        source_row = int(cache.source_rows[local_index])
        image = self._map_for(position)[source_row].astype(np.float32, copy=True)
        image *= 1.0 / 255.0
        image -= cache.mean
        image /= cache.std
        image = np.expand_dims(image, axis=0)
        label = int(cache.labels[local_index])
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


class I20CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        specifications = (
            (1, 64, (3, 1), (2, 1)),
            (64, 128, (1, 1), (1, 1)),
            (128, 256, (1, 1), (1, 1)),
        )
        layers: list[nn.Module] = []
        for in_channels, out_channels, stride, dilation in specifications:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(5, 3),
                        stride=stride,
                        padding=(2, 1),
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(),
                    nn.MaxPool2d((2, 1), ceil_mode=True),
                ]
            )
        layers.extend([nn.Flatten(), nn.Dropout(p=0.5)])
        self.features = nn.Sequential(*layers)
        flattened = 46_080
        if flattened != 46_080:
            raise AssertionError(f"Unexpected official I20 flatten size: {flattened}")
        self.fc = nn.Linear(flattened, 2)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(inputs))


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _split_indices(
    labels: np.ndarray,
    train_ratio: float,
    seed: int,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train-ratio must be between zero and one")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(labels), generator=generator).numpy()
    train_size = int(len(labels) * train_ratio)
    train_indices = permutation[:train_size].astype(np.int64, copy=False)
    validation_indices = permutation[train_size:].astype(np.int64, copy=False)
    digest = hashlib.sha256()
    digest.update(train_indices.tobytes())
    digest.update(validation_indices.tobytes())
    checksum = digest.hexdigest().upper()
    split_path = cache_dir / f"split_seed{seed}_ratio{train_ratio:.2f}.npz"
    temporary = split_path.with_suffix(".partial.npz")
    np.savez_compressed(
        temporary,
        train_indices=train_indices,
        validation_indices=validation_indices,
        checksum=np.asarray([checksum]),
    )
    os.replace(temporary, split_path)
    summary = {
        "seed": seed,
        "train_ratio": train_ratio,
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "train_up_rate": float(labels[train_indices].mean()),
        "validation_up_rate": float(labels[validation_indices].mean()),
        "split_sha256": checksum,
        "split_file": split_path.name,
    }
    return train_indices, validation_indices, summary


def _loader(
    subset: Subset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, Any] = {
        "dataset": subset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def _warmup(device: torch.device, batches: int, batch_size: int, seed: int) -> float:
    if batches <= 0:
        return 0.0
    _set_seeds(seed + 10_000)
    model = I20CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    inputs = torch.randn((batch_size, 1, *IMAGE_SHAPE), device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)
    started = time.perf_counter()
    for _ in range(batches):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(inputs), labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    del inputs, labels, optimizer, model
    torch.cuda.empty_cache()
    return elapsed


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float | None:
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator <= 0:
        return None
    return float((tp * tn - fp * fn) / math.sqrt(denominator))


def _run_phase(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    phase: str,
    progress_every: int,
    max_batches: int | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    expected_batches = len(loader)
    selected_batches = (
        expected_batches if max_batches is None else min(expected_batches, max_batches)
    )
    loss_sum = torch.zeros((), device=device)
    confusion = torch.zeros(4, dtype=torch.int64, device=device)
    sample_count = 0
    first_batch_seconds: float | None = None
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    iterator_started = time.perf_counter()
    iterator = iter(loader)
    data_wait_seconds = time.perf_counter() - iterator_started
    with context:
        for batch_number in range(1, selected_batches + 1):
            wait_started = time.perf_counter()
            inputs, labels = next(iterator)
            data_wait_seconds += time.perf_counter() - wait_started
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            if batch_number == 1:
                if tuple(inputs.shape[1:]) != (1, *IMAGE_SHAPE):
                    raise ValueError(f"Unexpected input shape: {tuple(inputs.shape)}")
                if not torch.isfinite(inputs).all().item():
                    raise ValueError("First input batch contains non-finite values")
                observed_labels = set(labels.detach().cpu().tolist())
                if not observed_labels.issubset({0, 1}):
                    raise ValueError(f"Unexpected labels: {observed_labels}")
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            if batch_number == 1 and not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite {phase} loss")
            if training:
                loss.backward()
                if batch_number == 1:
                    gradients_finite = all(
                        parameter.grad is None
                        or bool(torch.isfinite(parameter.grad).all().item())
                        for parameter in model.parameters()
                    )
                    if not gradients_finite:
                        raise FloatingPointError("Non-finite gradient in first batch")
                optimizer.step()
            predictions = outputs.argmax(dim=1)
            batch_samples = int(labels.numel())
            loss_sum += loss.detach() * batch_samples
            confusion += torch.bincount(labels * 2 + predictions, minlength=4)
            sample_count += batch_samples
            if batch_number == 1:
                torch.cuda.synchronize(device)
                first_batch_seconds = time.perf_counter() - started
            if progress_every > 0 and (
                batch_number % progress_every == 0 or batch_number == selected_batches
            ):
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                print(
                    f"{phase}: {batch_number:,}/{selected_batches:,} batches, "
                    f"{sample_count:,} samples, {sample_count / elapsed:,.1f} samples/s",
                    flush=True,
                )
            del inputs, labels, outputs, loss, predictions
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not torch.isfinite(loss_sum).item():
        raise FloatingPointError(f"Non-finite accumulated {phase} loss")
    if training and not all(
        bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()
    ):
        raise FloatingPointError("Non-finite model parameter after training phase")
    loss_value = float(loss_sum.item() / sample_count)
    tn, fp, fn, tp = [int(value) for value in confusion.cpu().tolist()]
    correct = tp + tn
    return {
        "phase": phase,
        "full_phase": max_batches is None or max_batches >= expected_batches,
        "expected_batches": expected_batches,
        "completed_batches": int(math.ceil(sample_count / loader.batch_size)),
        "samples": sample_count,
        "seconds": elapsed,
        "samples_per_second": sample_count / elapsed,
        "batches_per_second": (math.ceil(sample_count / loader.batch_size) / elapsed),
        "first_batch_seconds": first_batch_seconds,
        "data_wait_seconds": data_wait_seconds,
        "data_wait_share_of_wall_time": data_wait_seconds / elapsed,
        "loss": loss_value,
        "accuracy": correct / sample_count,
        "mcc": _mcc(tp=tp, tn=tn, fp=fp, fn=fn),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _format_duration(seconds: float) -> str:
    minutes, remaining = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}시간 {minutes}분 {remaining:.1f}초"
    return f"{minutes}분 {remaining:.1f}초"


def _write_report(result: dict[str, Any], path: Path) -> None:
    epoch = result["epoch"]
    train = epoch["train"]
    validation = epoch["validation"]
    full_epoch = epoch["full_epoch"]
    measured = epoch["seconds"]
    estimates = result["time_estimates"]
    status = "전체 1 epoch 완료" if full_epoch else "제한 배치 smoke test 완료"
    lines = [
        "# CNN I20 1-Epoch 벤치마크",
        "",
        f"실행일: {result['generated_at'][:10]}  ",
        f"상태: **{status}**",
        "",
        "## 결과",
        "",
        f"- GPU: `{result['environment']['device_name']}`",
        f"- PyTorch/CUDA: `{result['environment']['torch_version']}` / "
        f"`{result['environment']['torch_cuda_version']}`",
        f"- 표본: 학습 {result['split']['train_rows']:,}개, "
        f"검증 {result['split']['validation_rows']:,}개",
        f"- 배치: 학습 {train['completed_batches']:,}개, "
        f"검증 {validation['completed_batches']:,}개 (`batch_size={result['config']['batch_size']}`)",
        f"- 학습 시간: **{_format_duration(train['seconds'])}** "
        f"({train['samples_per_second']:,.1f} samples/s)",
        f"- 검증 시간: **{_format_duration(validation['seconds'])}** "
        f"({validation['samples_per_second']:,.1f} samples/s)",
        f"- 합계: **{_format_duration(measured)}**",
        f"- 최대 GPU 할당/예약 메모리: "
        f"{epoch['peak_gpu_memory_allocated_bytes'] / 2**30:.2f} / "
        f"{epoch['peak_gpu_memory_reserved_bytes'] / 2**30:.2f} GiB",
        "",
        "## 시간 환산",
        "",
        *(
            [
                f"- 모델 1개 10 epochs 단순 환산: {estimates['one_model_10_epochs_hours']:.2f}시간",
                f"- 모델 1개 50 epochs 상한 단순 환산: {estimates['one_model_50_epochs_hours']:.2f}시간",
                f"- 5개 앙상블 × 10 epochs 단순 환산: "
                f"{estimates['five_models_10_epochs_hours']:.2f}시간",
            ]
            if estimates is not None
            else ["제한 배치 smoke test에는 장기 학습시간 환산을 표시하지 않는다."]
        ),
        "",
        "전체 epoch 환산치는 CRC 캐시 검증 직후 측정한 epoch 시간이 계속된다는 단순 계산이다. 이후 epoch는 "
        "persistent worker와 시스템 상태에 따라 달라질 수 있고, 조기 종료·열 스로틀링·전원 모드에도 영향을 받는다. "
        "1 epoch의 loss와 accuracy는 성능 결론이 아니라 파이프라인 sanity check로만 본다.",
        "",
        "## 측정 정의",
        "",
        f"- 공식 I20 3-layer CNN, true FP32(TF32 off), Adam, CrossEntropy, "
        f"learning rate `{result['config']['learning_rate']}`",
        "- 안전하게 수정한 `Ret_month`/`EWMA_vol` 필터와 2000-12 누수 차단 적용",
        "- 1993–2000 적격 표본 전체를 고정 seed로 70/30 무작위 분할",
        "- 1 epoch는 학습 pass 1회와 검증 pass 1회이며, 추가 train 재평가·앙상블·OOS 예측은 포함하지 않음",
        f"- ZIP SHA-256 공식값 일치: `{result['archive']['sha256_matches_expected']}`; "
        "8개 `.dat`만 CRC 검증된 Git-ignored cache로 추출하여 memmap 사용",
        "- 공식 로더처럼 연도별 첫 50,000개 필터 전 이미지의 scalar mean/std로 정규화. "
        "이 방식은 같은 해 검증·미래월 분포를 포함하므로 성능 검정용 leakage-free 정규화가 아니며, "
        "이번 timing benchmark 밖에서 재사용하지 않음",
        "",
        "상세 재현 정보는 `outputs/cnn_epoch_benchmark/benchmark.json`에 있다.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.warmup_batches < 0:
        raise ValueError("warmup-batches cannot be negative")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    for name in ("max_train_batches", "max_validation_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if (args.max_train_batches is None) != (args.max_validation_batches is None):
        raise ValueError(
            "Smoke limits must be supplied together so one phase cannot accidentally "
            "run in full."
        )

    image_zip = args.image_zip.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    report = args.report.expanduser().resolve()
    generated_at = datetime.now().astimezone().isoformat()

    environment = _cuda_preflight()
    before_smi = _nvidia_smi_snapshot()
    setup_started = time.perf_counter()
    caches, cache_manifest = prepare_cache(
        image_zip, cache_dir, skip_sha256=args.skip_sha256
    )
    dataset = MultiYearImageDataset(caches)
    labels = np.concatenate([cache.labels for cache in caches])
    if len(dataset) != len(labels):
        raise AssertionError("Dataset and label lengths disagree")
    train_indices, validation_indices, split = _split_indices(
        labels, args.train_ratio, args.seed, cache_dir
    )
    train_loader = _loader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed + 1,
    )
    validation_loader = _loader(
        Subset(dataset, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 2,
    )
    setup_seconds = time.perf_counter() - setup_started

    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    warmup_seconds = _warmup(
        device, args.warmup_batches, args.batch_size, args.seed
    )

    _set_seeds(args.seed)
    model = I20CNN().to(device)
    parameter_count = _parameter_count(model)
    if parameter_count != EXPECTED_MODEL_PARAMETERS:
        raise AssertionError(
            f"Official model parameter count mismatch: {parameter_count} != "
            f"{EXPECTED_MODEL_PARAMETERS}"
        )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=0
    )
    loss_function = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    train_result = _run_phase(
        model=model,
        loader=train_loader,
        device=device,
        loss_function=loss_function,
        optimizer=optimizer,
        phase="train",
        progress_every=args.progress_every,
        max_batches=args.max_train_batches,
    )
    validation_result = _run_phase(
        model=model,
        loader=validation_loader,
        device=device,
        loss_function=loss_function,
        optimizer=None,
        phase="validation",
        progress_every=args.progress_every,
        max_batches=args.max_validation_batches,
    )
    epoch_seconds = train_result["seconds"] + validation_result["seconds"]
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    after_smi = _nvidia_smi_snapshot()
    full_epoch = bool(train_result["full_phase"] and validation_result["full_phase"])

    time_estimates = (
        {
            "one_model_10_epochs_hours": epoch_seconds * 10 / 3600,
            "one_model_50_epochs_hours": epoch_seconds * 50 / 3600,
            "five_models_10_epochs_hours": epoch_seconds * 50 / 3600,
            "five_models_50_epochs_hours": epoch_seconds * 250 / 3600,
            "assumption": (
                "linear scaling from the measured train+validation epoch after "
                "cache CRC verification; later epochs may differ"
            ),
        }
        if full_epoch
        else None
    )
    result: dict[str, Any] = {
        "generated_at": generated_at,
        "status": "full_epoch_complete" if full_epoch else "smoke_test_complete",
        "environment": environment,
        "nvidia_smi_before": before_smi,
        "nvidia_smi_after": after_smi,
        "archive": {
            "basename": image_zip.name,
            "sha256": cache_manifest["archive_sha256"],
            "sha256_matches_expected": cache_manifest[
                "archive_sha256_matches_expected"
            ],
        },
        "cache": cache_manifest,
        "split": split,
        "config": {
            "years": list(EXPECTED_YEARS),
            "target": "Ret_month > 0",
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "Adam",
            "weight_decay": 0,
            "loss": "CrossEntropyLoss",
            "precision": "float32",
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "warmup_batches": args.warmup_batches,
            "cudnn_benchmark": True,
            "deterministic_algorithms": False,
            "normalization_is_leakage_free": False,
            "model_parameters": parameter_count,
            "max_train_batches": args.max_train_batches,
            "max_validation_batches": args.max_validation_batches,
            "checkpoint_saved": False,
        },
        "setup": {
            "seconds_this_run": setup_seconds,
            "cache_extraction_seconds_this_run": cache_manifest[
                "total_extraction_seconds_this_run"
            ],
            "synthetic_cuda_warmup_seconds": warmup_seconds,
        },
        "epoch": {
            "full_epoch": full_epoch,
            "seconds": epoch_seconds,
            "train": train_result,
            "validation": validation_result,
            "peak_gpu_memory_allocated_bytes": peak_allocated,
            "peak_gpu_memory_reserved_bytes": peak_reserved,
        },
        "time_estimates": time_estimates,
        "interpretation_guardrail": (
            "One-epoch loss/accuracy/MCC are pipeline checks, not predictive-performance evidence."
        ),
    }
    _atomic_json(result, output)
    _write_report(result, report)
    dataset.close()
    print(f"Benchmark JSON: {output}", flush=True)
    print(f"Benchmark report: {report}", flush=True)
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
