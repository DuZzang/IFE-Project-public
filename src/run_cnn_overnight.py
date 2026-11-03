from __future__ import annotations

import argparse
import bisect
import contextlib
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from cnn_official_data import IMAGE_SHAPE
from run_cnn_epoch_benchmark import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_TRAIN_RATIO,
    EXPECTED_MODEL_PARAMETERS,
    I20CNN,
    YearCache,
    _atomic_json,
    _cuda_preflight,
    _mcc,
    _nvidia_smi_snapshot,
    _parameter_count,
    _set_seeds,
    _split_indices,
    prepare_cache,
)


DEFAULT_ENSEMBLE_SIZE = 5
DEFAULT_MAX_EPOCHS = 50
DEFAULT_PATIENCE = 2
DEFAULT_BASE_SEED = 42
NORMALIZATION_CHUNK_ROWS = 2_048


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(
        description=(
            "Train the official I20 CNN ensemble on 1993-2000 with train-only "
            "normalization, validation-loss early stopping, and resumable checkpoints."
        )
    )
    result.add_argument("--image-zip", type=Path, required=True)
    result.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "data" / "raw" / "cnn_i20_benchmark",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "cnn_overnight" / "i20_ret_month_1993_2000",
    )
    result.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "cnn_overnight_training.md",
    )
    result.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    result.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    result.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    result.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    result.add_argument("--ensemble-size", type=int, default=DEFAULT_ENSEMBLE_SIZE)
    result.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    result.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    result.add_argument("--num-workers", type=int, default=1)
    result.add_argument("--progress-every", type=int, default=250)
    result.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume compatible incomplete model checkpoints and skip completed models.",
    )
    result.add_argument(
        "--prevent-sleep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On Windows, prevent automatic idle sleep while this process is running.",
    )
    result.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Smoke-test only; must be supplied with --max-validation-batches.",
    )
    result.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help="Smoke-test only; must be supplied with --max-train-batches.",
    )
    result.add_argument("--skip-sha256", action="store_true")
    return result


def _now() -> str:
    return datetime.now().astimezone().isoformat()


@contextlib.contextmanager
def _exclusive_run_lock(path: Path) -> Iterator[None]:
    """Hold an OS-released lock so two trainers cannot share one output directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    locked = False
    try:
        handle = path.open("a+b", buffering=0)
        if path.stat().st_size == 0:
            os.lseek(handle.fileno(), 0, os.SEEK_SET)
            handle.write(b"0")
        os.lseek(handle.fileno(), 0, os.SEEK_SET)
        if platform.system() == "Windows":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(
                    f"Another CNN trainer is already using output directory: {path.parent}"
                ) from error
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError(
                    f"Another CNN trainer is already using output directory: {path.parent}"
                ) from error
        locked = True
        os.lseek(handle.fileno(), 0, os.SEEK_SET)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started_at={_now()}\n".encode("utf-8"))
        yield
    finally:
        if locked and handle is not None:
            os.lseek(handle.fileno(), 0, os.SEEK_SET)
            if platform.system() == "Windows":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        if handle is not None:
            handle.close()


def _write_failure_state(state_path: Path, error: BaseException) -> None:
    previous: dict[str, Any] = {}
    if state_path.is_file():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except (OSError, json.JSONDecodeError):
            pass
    previous.update(
        {
            "status": "failed",
            "pid": os.getpid(),
            "failed_at": _now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    _atomic_json(previous, state_path)


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch-size": args.batch_size,
        "learning-rate": args.learning_rate,
        "ensemble-size": args.ensemble_size,
        "max-epochs": args.max_epochs,
        "patience": args.patience,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train-ratio must be between zero and one")
    smoke_values = (args.max_train_batches, args.max_validation_batches)
    if (smoke_values[0] is None) != (smoke_values[1] is None):
        raise ValueError("Both smoke-test batch limits must be supplied together")
    if any(value is not None and value <= 0 for value in smoke_values):
        raise ValueError("Smoke-test batch limits must be positive")


@contextlib.contextmanager
def _windows_sleep_guard(enabled: bool) -> Iterator[bool]:
    """Prevent automatic idle sleep for the process lifetime, then restore defaults."""

    active = False
    if enabled and platform.system() == "Windows":
        es_continuous = 0x80000000
        es_system_required = 0x00000001
        result = ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | es_system_required
        )
        if result == 0:
            raise OSError("SetThreadExecutionState failed")
        active = True
    try:
        yield active
    finally:
        if active:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


class FixedNormalizationImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """I20 images normalized only with statistics from the fixed training split."""

    def __init__(
        self,
        caches: Sequence[YearCache],
        *,
        mean: float,
        std: float,
    ) -> None:
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid normalization statistics: mean={mean}, std={std}")
        self.caches = list(caches)
        self.mean = float(mean)
        self.std = float(std)
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
        image -= self.mean
        image /= self.std
        return (
            torch.from_numpy(np.expand_dims(image, axis=0)),
            torch.tensor(int(cache.labels[local_index]), dtype=torch.long),
        )


def _training_source_rows_by_year(
    caches: Sequence[YearCache], train_indices: np.ndarray
) -> list[np.ndarray]:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if train_indices.ndim != 1:
        raise ValueError("train_indices must be one-dimensional")
    result: list[np.ndarray] = []
    offset = 0
    for cache in caches:
        upper = offset + cache.eligible_count
        selected = train_indices[(train_indices >= offset) & (train_indices < upper)]
        local = selected - offset
        source_rows = np.sort(cache.source_rows[local].astype(np.int64, copy=False))
        result.append(source_rows)
        offset = upper
    if sum(len(rows) for rows in result) != len(train_indices):
        raise ValueError("Some training indices fall outside the concatenated dataset")
    return result


def _normalization_cache_path(cache_dir: Path, split_sha256: str) -> Path:
    return cache_dir / f"train_normalization_{split_sha256[:16].lower()}.json"


def compute_train_normalization(
    caches: Sequence[YearCache],
    train_indices: np.ndarray,
    *,
    split_sha256: str,
    cache_dir: Path,
    chunk_rows: int = NORMALIZATION_CHUNK_ROWS,
) -> dict[str, Any]:
    """Compute an exact global pixel mean/std using training rows only."""

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    destination = _normalization_cache_path(cache_dir, split_sha256)
    if destination.is_file():
        cached = json.loads(destination.read_text(encoding="utf-8"))
        if (
            cached.get("split_sha256") == split_sha256
            and cached.get("train_rows") == int(len(train_indices))
            and cached.get("image_shape") == list(IMAGE_SHAPE)
            and cached.get("method") == "exact_train_rows_global_scalar"
        ):
            return cached

    source_rows_by_year = _training_source_rows_by_year(caches, train_indices)
    pixel_sum = 0.0
    pixel_square_sum = 0.0
    pixel_count = 0
    rows_seen = 0
    started = time.perf_counter()
    per_year: list[dict[str, Any]] = []
    for cache, source_rows in zip(caches, source_rows_by_year, strict=True):
        image_map = np.memmap(
            cache.image_path,
            dtype=np.uint8,
            mode="r",
            shape=(cache.image_count, *IMAGE_SHAPE),
        )
        year_sum = 0.0
        year_square_sum = 0.0
        for start in range(0, len(source_rows), chunk_rows):
            rows = source_rows[start : start + chunk_rows]
            values = np.asarray(image_map[rows], dtype=np.float64)
            year_sum += float(values.sum(dtype=np.float64))
            year_square_sum += float(np.square(values).sum(dtype=np.float64))
        memory_map = getattr(image_map, "_mmap", None)
        if memory_map is not None:
            memory_map.close()
        year_pixels = int(len(source_rows) * math.prod(IMAGE_SHAPE))
        pixel_sum += year_sum
        pixel_square_sum += year_square_sum
        pixel_count += year_pixels
        rows_seen += len(source_rows)
        per_year.append({"year": cache.year, "train_rows": int(len(source_rows))})
        print(
            f"normalization: {cache.year} complete ({len(source_rows):,} train rows)",
            flush=True,
        )

    if rows_seen != len(train_indices) or pixel_count <= 0:
        raise AssertionError("Training-only normalization row count mismatch")
    mean_raw = pixel_sum / pixel_count
    variance_raw = max(pixel_square_sum / pixel_count - mean_raw * mean_raw, 0.0)
    std_raw = math.sqrt(variance_raw)
    result = {
        "generated_at": _now(),
        "method": "exact_train_rows_global_scalar",
        "split_sha256": split_sha256,
        "train_rows": int(rows_seen),
        "pixel_count": int(pixel_count),
        "image_shape": list(IMAGE_SHAPE),
        "mean": mean_raw / 255.0,
        "std": std_raw / 255.0,
        "seconds": time.perf_counter() - started,
        "per_year": per_year,
    }
    if not math.isfinite(result["mean"]) or not math.isfinite(result["std"]):
        raise FloatingPointError("Non-finite normalization statistics")
    if result["std"] <= 0:
        raise ValueError("Training-only normalization standard deviation is not positive")
    _atomic_json(result, destination)
    return result


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
        # Recreate workers each epoch. With persistent workers, a restarted loader
        # consumes its shared generator differently from an uninterrupted loader,
        # shifting RandomSampler order after resume.
        kwargs.update(persistent_workers=False, prefetch_factor=2)
    return DataLoader(**kwargs)


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
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        iterator = iter(loader)
        for batch_number in range(1, selected_batches + 1):
            inputs, labels = next(iterator)
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite {phase} loss")
            if training:
                loss.backward()
                optimizer.step()
            predictions = outputs.argmax(dim=1)
            batch_samples = int(labels.numel())
            loss_sum += loss.detach() * batch_samples
            confusion += torch.bincount(labels * 2 + predictions, minlength=4)
            sample_count += batch_samples
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
    if sample_count <= 0:
        raise RuntimeError(f"{phase} completed no samples")
    tn, fp, fn, tp = [int(value) for value in confusion.cpu().tolist()]
    return {
        "phase": phase,
        "full_phase": selected_batches == expected_batches,
        "expected_batches": expected_batches,
        "completed_batches": selected_batches,
        "samples": sample_count,
        "seconds": elapsed,
        "samples_per_second": sample_count / elapsed,
        "loss": float(loss_sum.item() / sample_count),
        "accuracy": (tp + tn) / sample_count,
        "mcc": _mcc(tp=tp, tn=tn, fp=fp, fn=fn),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _torch_save_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _config_signature(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _model_seed(base_seed: int, model_index: int) -> int:
    return base_seed + model_index


def _set_reproducibility() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _last_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    stale_epochs: int,
    history: list[dict[str, Any]],
    signature: str,
    train_loader: DataLoader,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "saved_at": _now(),
        "config_signature": signature,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "stale_epochs": stale_epochs,
        "history": history,
        "model_state_dict": _state_dict_cpu(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "train_loader_generator_state": train_loader.generator.get_state(),
    }


def _restore_checkpoint(
    *,
    checkpoint: dict[str, Any],
    signature: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
) -> tuple[int, int, float, int, list[dict[str, Any]]]:
    if checkpoint.get("config_signature") != signature:
        raise ValueError("Existing checkpoint configuration does not match this run")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    train_loader.generator.set_state(checkpoint["train_loader_generator_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    start_epoch = int(checkpoint["epoch"]) + 1
    return (
        start_epoch,
        int(checkpoint["best_epoch"]),
        float(checkpoint["best_validation_loss"]),
        int(checkpoint["stale_epochs"]),
        list(checkpoint["history"]),
    )


def _close_loader(loader: DataLoader | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


def _run_one_model(
    *,
    model_index: int,
    model_seed: int,
    dataset: Dataset,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    shared_config: dict[str, Any],
    output_dir: Path,
    state_path: Path,
    started_at: str,
) -> dict[str, Any]:
    model_dir = output_dir / f"model_{model_index:02d}_seed{model_seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    complete_path = model_dir / "complete.json"
    last_path = model_dir / "last.pt"
    best_path = model_dir / "best.pt"
    history_path = model_dir / "history.json"

    model_config = {**shared_config, "model_index": model_index, "model_seed": model_seed}
    signature = _config_signature(model_config)
    if args.resume and complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("config_signature") != signature:
            raise ValueError(f"Completed model {model_index} has an incompatible configuration")
        print(f"model {model_index}: compatible completed model found; skipping", flush=True)
        return complete

    _set_seeds(model_seed)
    model = I20CNN().to(device)
    if _parameter_count(model) != EXPECTED_MODEL_PARAMETERS:
        raise AssertionError("I20 model parameter count changed")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
    loss_function = nn.CrossEntropyLoss()
    train_loader = _loader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=model_seed + 100_000,
    )
    validation_loader = _loader(
        Subset(dataset, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=model_seed + 200_000,
    )
    start_epoch = 0
    best_epoch = -1
    best_validation_loss = math.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    if args.resume and last_path.is_file():
        # RNG and DataLoader generator states must remain CPU ByteTensors. Loading
        # the whole checkpoint onto CUDA makes set_rng_state()/Generator.set_state()
        # fail before training can resume. Optimizer state is moved to each CUDA
        # parameter's device by optimizer.load_state_dict().
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        (
            start_epoch,
            best_epoch,
            best_validation_loss,
            stale_epochs,
            history,
        ) = _restore_checkpoint(
            checkpoint=checkpoint,
            signature=signature,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
        )
        print(f"model {model_index}: resuming at epoch {start_epoch}", flush=True)
    elif last_path.exists() or best_path.exists() or history_path.exists():
        raise FileExistsError(
            f"Model {model_index} has existing artifacts; use --resume or a new output directory"
        )

    model_started = time.perf_counter()
    stop_reason = "early_stopping" if stale_epochs >= args.patience else "max_epochs"
    try:
        if stale_epochs >= args.patience:
            print(
                f"model {model_index}: checkpoint already satisfies early stopping; "
                "finalizing without another epoch",
                flush=True,
            )
        for epoch in range(start_epoch, args.max_epochs):
            if stale_epochs >= args.patience:
                break
            epoch_number = epoch + 1
            print(
                f"model {model_index + 1}/{args.ensemble_size}, "
                f"epoch {epoch_number}/{args.max_epochs}",
                flush=True,
            )
            _atomic_json(
                {
                    "status": "running",
                    "phase": "train",
                    "pid": os.getpid(),
                    "started_at": started_at,
                    "updated_at": _now(),
                    "model_index": model_index,
                    "model_seed": model_seed,
                    "epoch_number": epoch_number,
                    "max_epochs": args.max_epochs,
                    "best_epoch_number": best_epoch + 1 if best_epoch >= 0 else None,
                    "best_validation_loss": (
                        best_validation_loss if math.isfinite(best_validation_loss) else None
                    ),
                    "stale_epochs": stale_epochs,
                },
                state_path,
            )
            train_result = _run_phase(
                model=model,
                loader=train_loader,
                device=device,
                loss_function=loss_function,
                optimizer=optimizer,
                phase=f"model{model_index:02d}/epoch{epoch_number:02d}/train",
                progress_every=args.progress_every,
                max_batches=args.max_train_batches,
            )
            _atomic_json(
                {
                    "status": "running",
                    "phase": "validation",
                    "pid": os.getpid(),
                    "started_at": started_at,
                    "updated_at": _now(),
                    "model_index": model_index,
                    "model_seed": model_seed,
                    "epoch_number": epoch_number,
                    "max_epochs": args.max_epochs,
                    "best_epoch_number": best_epoch + 1 if best_epoch >= 0 else None,
                    "best_validation_loss": (
                        best_validation_loss if math.isfinite(best_validation_loss) else None
                    ),
                    "stale_epochs": stale_epochs,
                },
                state_path,
            )
            validation_result = _run_phase(
                model=model,
                loader=validation_loader,
                device=device,
                loss_function=loss_function,
                optimizer=None,
                phase=f"model{model_index:02d}/epoch{epoch_number:02d}/validation",
                progress_every=args.progress_every,
                max_batches=args.max_validation_batches,
            )
            improved = validation_result["loss"] < best_validation_loss
            if improved:
                best_epoch = epoch
                best_validation_loss = float(validation_result["loss"])
                stale_epochs = 0
                _torch_save_atomic(
                    {
                        "format_version": 1,
                        "saved_at": _now(),
                        "config": model_config,
                        "config_signature": signature,
                        "epoch": epoch,
                        "validation": validation_result,
                        "model_state_dict": _state_dict_cpu(model),
                    },
                    best_path,
                )
            else:
                stale_epochs += 1
            epoch_record = {
                "epoch": epoch,
                "epoch_number": epoch_number,
                "improved": improved,
                "stale_epochs": stale_epochs,
                "train": train_result,
                "validation": validation_result,
            }
            history.append(epoch_record)
            _atomic_json({"config": model_config, "epochs": history}, history_path)
            _torch_save_atomic(
                _last_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_validation_loss=best_validation_loss,
                    stale_epochs=stale_epochs,
                    history=history,
                    signature=signature,
                    train_loader=train_loader,
                ),
                last_path,
            )
            _atomic_json(
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "started_at": started_at,
                    "updated_at": _now(),
                    "model_index": model_index,
                    "model_seed": model_seed,
                    "epoch_number": epoch_number,
                    "max_epochs": args.max_epochs,
                    "best_epoch_number": best_epoch + 1,
                    "best_validation_loss": best_validation_loss,
                    "stale_epochs": stale_epochs,
                },
                state_path,
            )
            print(
                f"epoch {epoch_number}: train loss={train_result['loss']:.6f}, "
                f"validation loss={validation_result['loss']:.6f}, "
                f"validation MCC={validation_result['mcc']}, improved={improved}",
                flush=True,
            )
            if stale_epochs >= args.patience:
                stop_reason = "early_stopping"
                print(
                    f"model {model_index}: early stopping after {stale_epochs} "
                    "non-improving epochs",
                    flush=True,
                )
                break

        if best_epoch < 0 or not best_path.is_file():
            raise RuntimeError(f"Model {model_index} never produced a valid best checkpoint")
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        result = {
            "status": "complete",
            "completed_at": _now(),
            "config": model_config,
            "config_signature": signature,
            "model_index": model_index,
            "model_seed": model_seed,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_epoch_number": best_epoch + 1,
            "best_validation": best["validation"],
            "stop_reason": stop_reason,
            "seconds": time.perf_counter() - model_started,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "history_file": str(history_path),
        }
        _atomic_json(result, complete_path)
        return result
    finally:
        _close_loader(train_loader)
        _close_loader(validation_loader)
        del model, optimizer, train_loader, validation_loader
        gc.collect()
        torch.cuda.empty_cache()


def _write_report(summary: dict[str, Any], path: Path) -> None:
    models = summary["models"]
    lines = [
        "# CNN I20 Overnight Training",
        "",
        f"- 상태: **{summary['status']}**",
        f"- 시작: `{summary['started_at']}`",
        f"- 종료: `{summary['completed_at']}`",
        f"- GPU: `{summary['environment']['device_name']}`",
        f"- 표본: train {summary['split']['train_rows']:,}, "
        f"validation {summary['split']['validation_rows']:,}",
        f"- 모델: {len(models)}개 ensemble, 최대 {summary['config']['max_epochs']} epochs, "
        f"patience {summary['config']['patience']}",
        "- 목표: `Ret_month > 0`",
        "- 정규화: 고정 train split 픽셀만 사용한 global scalar mean/std",
        "",
        "## 모델별 최선의 검증 결과",
        "",
        "| Model | Seed | Epochs | Best epoch | Loss | Accuracy | MCC | Stop |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in models:
        validation = item["best_validation"]
        mcc = validation["mcc"]
        mcc_text = "NA" if mcc is None else f"{mcc:.6f}"
        lines.append(
            f"| {item['model_index']} | {item['model_seed']} | "
            f"{item['epochs_completed']} | {item['best_epoch_number']} | "
            f"{validation['loss']:.6f} | {validation['accuracy']:.6f} | "
            f"{mcc_text} | {item['stop_reason']} |"
        )
    lines.extend(
        [
            "",
            "검증 결과는 모델 선택과 파이프라인 점검용이다. 최종 예측력 평가는 이 모델들을 "
            "고정한 뒤 2001–2019 OOS 표본에서 월별 포트폴리오로 수행해야 한다.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    image_zip = args.image_zip.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "run_state.json"
    summary_path = output_dir / "summary.json"
    report_path = args.report.expanduser().resolve()

    started_at = _now()
    wall_started = time.perf_counter()
    environment = _cuda_preflight()
    _set_reproducibility()
    before_smi = _nvidia_smi_snapshot()
    _atomic_json(
        {
            "status": "preparing",
            "pid": os.getpid(),
            "started_at": started_at,
            "updated_at": _now(),
        },
        state_path,
    )

    caches, cache_manifest = prepare_cache(
        image_zip, cache_dir, skip_sha256=args.skip_sha256
    )
    labels = np.concatenate([cache.labels for cache in caches])
    train_indices, validation_indices, split = _split_indices(
        labels, args.train_ratio, args.base_seed, cache_dir
    )
    normalization = compute_train_normalization(
        caches,
        train_indices,
        split_sha256=split["split_sha256"],
        cache_dir=cache_dir,
    )
    dataset = FixedNormalizationImageDataset(
        caches, mean=normalization["mean"], std=normalization["std"]
    )
    if len(dataset) != len(labels):
        raise AssertionError("Dataset and label lengths disagree")

    shared_config = {
        "archive_sha256": cache_manifest["archive_sha256"],
        "archive_sha256_matches_expected": cache_manifest[
            "archive_sha256_matches_expected"
        ],
        "years": list(range(1993, 2001)),
        "target": "Ret_month > 0",
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "weight_decay": 0,
        "loss": "CrossEntropyLoss",
        "dropout": 0.5,
        "train_ratio": args.train_ratio,
        "split_seed": args.base_seed,
        "split_sha256": split["split_sha256"],
        "normalization": normalization,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "early_stopping_metric": "validation_loss",
        "ensemble_size": args.ensemble_size,
        "precision": "float32",
        "tf32": False,
        "deterministic_algorithms": True,
        "model_parameters": EXPECTED_MODEL_PARAMETERS,
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
    }

    results: list[dict[str, Any]] = []
    try:
        for model_index in range(args.ensemble_size):
            model_seed = _model_seed(args.base_seed, model_index)
            results.append(
                _run_one_model(
                    model_index=model_index,
                    model_seed=model_seed,
                    dataset=dataset,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    device=torch.device("cuda:0"),
                    args=args,
                    shared_config=shared_config,
                    output_dir=output_dir,
                    state_path=state_path,
                    started_at=started_at,
                )
            )
        completed_at = _now()
        summary = {
            "status": "complete",
            "started_at": started_at,
            "completed_at": completed_at,
            "seconds": time.perf_counter() - wall_started,
            "environment": environment,
            "nvidia_smi_before": before_smi,
            "nvidia_smi_after": _nvidia_smi_snapshot(),
            "archive": {
                "basename": image_zip.name,
                "sha256": cache_manifest["archive_sha256"],
                "sha256_matches_expected": cache_manifest[
                    "archive_sha256_matches_expected"
                ],
            },
            "split": split,
            "normalization": normalization,
            "config": shared_config,
            "models": results,
            "next_step": "Freeze the ensemble and score the 2001-2019 OOS images.",
        }
        _atomic_json(summary, summary_path)
        _atomic_json(
            {
                "status": "complete",
                "pid": os.getpid(),
                "started_at": started_at,
                "completed_at": completed_at,
                "models_completed": len(results),
                "summary": str(summary_path),
            },
            state_path,
        )
        _write_report(summary, report_path)
        print(f"Training summary: {summary_path}", flush=True)
        print(f"Training report: {report_path}", flush=True)
        return summary
    finally:
        dataset.close()


def main() -> None:
    args = parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "run_state.json"
    with _exclusive_run_lock(output_dir / "run.lock"):
        launched_at = _now()
        _atomic_json(
            {
                "status": "launching",
                "pid": os.getpid(),
                "started_at": launched_at,
                "updated_at": launched_at,
            },
            state_path,
        )
        try:
            with _windows_sleep_guard(args.prevent_sleep) as sleep_guard_active:
                print(f"Windows sleep guard active: {sleep_guard_active}", flush=True)
                run(args)
        except BaseException as error:
            _write_failure_state(state_path, error)
            raise


if __name__ == "__main__":
    main()
