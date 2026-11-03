from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "CNN training environment is optional")
class CnnEpochBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        src = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(src))
        import run_cnn_epoch_benchmark as benchmark

        cls.benchmark = benchmark

    def test_official_model_shape_and_parameter_count(self) -> None:
        import torch

        model = self.benchmark.I20CNN()
        output = model(torch.zeros(2, 1, 64, 60))
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(
            self.benchmark._parameter_count(model),
            self.benchmark.EXPECTED_MODEL_PARAMETERS,
        )
        self.assertEqual(model.features[0].dilation, (2, 1))
        self.assertEqual(model.features[0].stride, (3, 1))
        self.assertIsInstance(model.features[1], torch.nn.BatchNorm2d)
        self.assertIsInstance(model.features[2], torch.nn.LeakyReLU)
        self.assertEqual(model.features[3].kernel_size, (2, 1))

    def test_split_is_deterministic_and_complete(self) -> None:
        labels = np.asarray([0, 1] * 50, dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            first = self.benchmark._split_indices(
                labels, 0.7, 42, Path(directory)
            )
            second = self.benchmark._split_indices(
                labels, 0.7, 42, Path(directory)
            )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(len(first[0]), 70)
        self.assertEqual(len(first[1]), 30)
        self.assertEqual(len(set(first[0]).intersection(first[1])), 0)
        self.assertEqual(first[2]["split_sha256"], second[2]["split_sha256"])

    def test_dataset_preserves_source_row_and_normalization(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "images.dat"
            images = np.zeros((3, 64, 60), dtype=np.uint8)
            images[2].fill(255)
            images.tofile(image_path)
            cache = self.benchmark.YearCache(
                year=1993,
                image_path=image_path,
                image_count=3,
                source_rows=np.asarray([2], dtype=np.int64),
                labels=np.asarray([1], dtype=np.int64),
                mean=0.5,
                std=0.25,
                extraction_seconds=0.0,
                cache_reused=False,
                image_crc32=0,
            )
            dataset = self.benchmark.MultiYearImageDataset([cache])
            image, label = dataset[0]
            dataset.close()
        self.assertEqual(tuple(image.shape), (1, 64, 60))
        self.assertTrue(torch.allclose(image, torch.full_like(image, 2.0)))
        self.assertEqual(int(label.item()), 1)


if __name__ == "__main__":
    unittest.main()
