from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
else:
    CUDA_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "CNN training environment is optional")
class CnnOvernightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        src = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(src))
        import run_cnn_overnight as overnight

        cls.overnight = overnight

    def _cache(
        self,
        directory: Path,
        *,
        year: int,
        values: list[int],
        labels: list[int] | None = None,
    ):
        image_path = directory / f"{year}_images.dat"
        images = np.stack(
            [np.full((64, 60), value, dtype=np.uint8) for value in values]
        )
        images.tofile(image_path)
        if labels is None:
            labels = [value % 2 for value in values]
        return self.overnight.YearCache(
            year=year,
            image_path=image_path,
            image_count=len(values),
            source_rows=np.arange(len(values), dtype=np.int64),
            labels=np.asarray(labels, dtype=np.int64),
            mean=0.0,
            std=1.0,
            extraction_seconds=0.0,
            cache_reused=False,
            image_crc32=0,
        )

    def test_parser_uses_paper_style_overnight_defaults(self) -> None:
        args = self.overnight.parser().parse_args(["--image-zip", "images.zip"])
        self.assertEqual(args.ensemble_size, 5)
        self.assertEqual(args.max_epochs, 50)
        self.assertEqual(args.patience, 2)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertTrue(args.resume)
        self.assertTrue(args.prevent_sleep)

    def test_training_rows_are_mapped_across_years(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            caches = [
                self._cache(root, year=1993, values=[0, 1, 2]),
                self._cache(root, year=1994, values=[3, 4]),
            ]
            rows = self.overnight._training_source_rows_by_year(
                caches, np.asarray([4, 0, 2], dtype=np.int64)
            )
        np.testing.assert_array_equal(rows[0], np.asarray([0, 2]))
        np.testing.assert_array_equal(rows[1], np.asarray([1]))

    def test_normalization_uses_only_training_rows_and_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = self._cache(root, year=1993, values=[0, 255, 128])
            result = self.overnight.compute_train_normalization(
                [cache],
                np.asarray([0, 1], dtype=np.int64),
                split_sha256="A" * 64,
                cache_dir=root,
                chunk_rows=1,
            )
            self.assertAlmostEqual(result["mean"], 0.5, places=12)
            self.assertAlmostEqual(result["std"], 0.5, places=12)
            self.assertEqual(result["train_rows"], 2)
            cache_file = root / "train_normalization_aaaaaaaaaaaaaaaa.json"
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            self.assertEqual(cached["method"], "exact_train_rows_global_scalar")
            second = self.overnight.compute_train_normalization(
                [cache],
                np.asarray([0, 1], dtype=np.int64),
                split_sha256="A" * 64,
                cache_dir=root,
                chunk_rows=1,
            )
            self.assertEqual(second, cached)

    def test_dataset_applies_fixed_training_normalization(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = self._cache(root, year=1993, values=[255], labels=[1])
            dataset = self.overnight.FixedNormalizationImageDataset(
                [cache], mean=0.5, std=0.25
            )
            image, label = dataset[0]
            dataset.close()
        self.assertTrue(torch.allclose(image, torch.full_like(image, 2.0)))
        self.assertEqual(int(label.item()), 1)

    def test_config_signature_is_order_independent(self) -> None:
        first = self.overnight._config_signature({"a": 1, "b": 2})
        second = self.overnight._config_signature({"b": 2, "a": 1})
        self.assertEqual(first, second)

    def test_loader_generator_resume_matches_next_epoch_shuffle(self) -> None:
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.arange(20))
        uninterrupted = self.overnight._loader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=1,
            seed=123,
        )
        list(iter(uninterrupted))
        saved_state = uninterrupted.generator.get_state()
        expected = torch.cat([batch[0] for batch in uninterrupted]).tolist()
        self.overnight._close_loader(uninterrupted)

        resumed = self.overnight._loader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=1,
            seed=999,
        )
        resumed.generator.set_state(saved_state)
        actual = torch.cat([batch[0] for batch in resumed]).tolist()
        self.overnight._close_loader(resumed)
        self.assertEqual(actual, expected)

    def test_argument_validation_requires_paired_smoke_limits(self) -> None:
        args = Namespace(
            batch_size=128,
            learning_rate=1e-4,
            ensemble_size=5,
            max_epochs=50,
            patience=2,
            num_workers=1,
            train_ratio=0.7,
            max_train_batches=1,
            max_validation_batches=None,
        )
        with self.assertRaisesRegex(ValueError, "Both smoke-test"):
            self.overnight._validate_args(args)

    def test_failure_state_preserves_progress_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "run_state.json"
            state_path.write_text(
                json.dumps({"status": "preparing", "started_at": "original", "model_index": 2}),
                encoding="utf-8",
            )
            self.overnight._write_failure_state(state_path, RuntimeError("boom"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["started_at"], "original")
        self.assertEqual(state["model_index"], 2)
        self.assertEqual(state["error_type"], "RuntimeError")
        self.assertEqual(state["error"], "boom")

    def test_output_directory_lock_rejects_second_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.lock"
            for _ in range(3):
                with self.overnight._exclusive_run_lock(path):
                    with self.assertRaisesRegex(RuntimeError, "Another CNN trainer"):
                        with self.overnight._exclusive_run_lock(path):
                            self.fail("Second trainer unexpectedly acquired the lock")

    @unittest.skipUnless(CUDA_AVAILABLE, "Checkpoint round-trip requires CUDA")
    def test_checkpoint_rng_states_round_trip_through_cpu_load(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda:0")
        model = torch.nn.Linear(2, 2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        inputs = torch.ones((2, 2), device=device)
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        loader = DataLoader(
            TensorDataset(torch.ones((4, 2)), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(123),
        )
        next(iter(loader))
        payload = self.overnight._last_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=2,
            best_epoch=1,
            best_validation_loss=0.5,
            stale_epochs=1,
            history=[{"epoch": 2}],
            signature="ABC",
            train_loader=loader,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save(payload, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)

        restored_model = torch.nn.Linear(2, 2).to(device)
        restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-4)
        restored_loader = DataLoader(
            TensorDataset(torch.ones((4, 2)), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(999),
        )
        restored = self.overnight._restore_checkpoint(
            checkpoint=loaded,
            signature="ABC",
            model=restored_model,
            optimizer=restored_optimizer,
            train_loader=restored_loader,
        )
        self.assertEqual(restored[:4], (3, 1, 0.5, 1))
        self.assertEqual(restored[4], [{"epoch": 2}])
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(restored_optimizer.state)
        for state in restored_optimizer.state.values():
            self.assertEqual(state["exp_avg"].device.type, "cuda")
            self.assertEqual(state["exp_avg_sq"].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
