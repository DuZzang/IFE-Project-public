from __future__ import annotations

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnn_official_data import (
    IMAGE_BYTES,
    IMAGE_SHAPE,
    catalog_archive,
    iter_filtered_year_image_batches,
    iter_year_image_batches,
    match_by_exchange,
    match_universes,
    prepare_classification_labels,
    read_year_labels,
    validate_label_partition,
    verify_official_archive_hash,
)


def label_frame(year: int, rows: int) -> pd.DataFrame:
    dates = pd.date_range(f"{year}-01-31", periods=rows, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "StockID": [str(10_000 + index) for index in range(rows)],
            "MarketCap": np.arange(rows, dtype=np.float32) + 1,
            "Ret_5d": np.linspace(-0.01, 0.01, rows),
            "Ret_20d": np.linspace(-0.02, 0.02, rows),
            "Ret_60d": np.linspace(-0.03, 0.03, rows),
            "Ret_month": np.linspace(-0.04, 0.04, rows),
            "EWMA_vol": np.linspace(0.01, 0.02, rows),
        }
    )


def write_archive(path: Path, *, year: int = 2000, images: int = 3, labels: int = 3) -> None:
    feather = io.BytesIO()
    label_frame(year, labels).to_feather(feather)
    image_payload = np.arange(images * IMAGE_BYTES, dtype=np.uint8).tobytes()
    prefix = f"monthly_20d/20d_month_has_vb_[20]_ma_{year}"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{prefix}_images.dat", image_payload)
        archive.writestr(f"{prefix}_labels_w_delay.feather", feather.getvalue())
        archive.writestr("__MACOSX/monthly_20d/._ignored", b"metadata")


class OfficialCnnDataTests(unittest.TestCase):
    def test_archive_hash_mismatch_stops_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not_official.zip"
            path.write_bytes(b"not the official archive")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_official_archive_hash(path, skip=False)
            self.assertEqual(
                verify_official_archive_hash(path, skip=True), (None, None)
            )

    def test_catalog_reads_pair_and_ignores_macos_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.zip"
            write_archive(path)
            catalog = catalog_archive(path)
            self.assertEqual([item.year for item in catalog], [2000])
            self.assertEqual(catalog[0].image_count, 3)
            self.assertEqual(read_year_labels(path, 2000, catalog=catalog).shape[0], 3)

    def test_image_batches_preserve_offsets_shape_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.zip"
            write_archive(path)
            batches = list(iter_year_image_batches(path, 2000, batch_size=2))
            self.assertEqual([offset for offset, _ in batches], [0, 2])
            self.assertEqual(batches[0][1].shape, (2, *IMAGE_SHAPE))
            self.assertEqual(batches[1][1].shape, (1, *IMAGE_SHAPE))
            reconstructed = np.concatenate([batch for _, batch in batches]).ravel()
            expected = np.arange(3 * IMAGE_BYTES, dtype=np.uint8)
            np.testing.assert_array_equal(reconstructed, expected)

    def test_filtered_batches_keep_source_row_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.zip"
            write_archive(path)
            batches = list(
                iter_filtered_year_image_batches(
                    path, 2000, [True, False, True], batch_size=2
                )
            )
            rows = np.concatenate([source_rows for source_rows, _ in batches])
            images = np.concatenate([batch for _, batch in batches])
            np.testing.assert_array_equal(rows, [0, 2])
            expected = np.arange(3 * IMAGE_BYTES, dtype=np.uint8).reshape(
                3, *IMAGE_SHAPE
            )[[0, 2]]
            np.testing.assert_array_equal(images, expected)

    def test_catalog_rejects_partial_image_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.zip"
            feather = io.BytesIO()
            label_frame(2000, 1).to_feather(feather)
            prefix = "monthly_20d/20d_month_has_vb_[20]_ma_2000"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(f"{prefix}_images.dat", b"0" * (IMAGE_BYTES + 1))
                archive.writestr(
                    f"{prefix}_labels_w_delay.feather", feather.getvalue()
                )
            with self.assertRaisesRegex(ValueError, "not divisible"):
                catalog_archive(path)

    def test_validation_rejects_image_label_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.zip"
            write_archive(path, images=2, labels=3)
            catalog = catalog_archive(path)
            labels = read_year_labels(path, 2000, catalog=catalog)
            with self.assertRaisesRegex(ValueError, "image_count=2 label_rows=3"):
                validate_label_partition(catalog[0], labels)

    def test_model_label_adapter_filters_before_classification(self) -> None:
        labels = label_frame(2000, 4)
        labels.loc[0, "Ret_month"] = np.nan
        labels.loc[1, "EWMA_vol"] = np.nan
        labels.loc[2, "EWMA_vol"] = 0.0
        labels.loc[3, "Ret_month"] = 0.02
        labels.loc[3, "MarketCap"] = -50.0
        prepared, keep_mask, audit = prepare_classification_labels(labels)
        np.testing.assert_array_equal(keep_mask, [False, False, False, True])
        self.assertEqual(prepared["source_row"].tolist(), [3])
        self.assertEqual(prepared["label"].tolist(), [1])
        self.assertEqual(prepared["MarketCap"].tolist(), [50.0])
        self.assertEqual(audit["kept_rows"], 1)
        self.assertEqual(
            audit["negative_market_cap_rows_converted_to_absolute"], 1
        )

    def test_model_label_adapter_applies_exclusive_is_oos_cutoff(self) -> None:
        labels = label_frame(2000, 4)
        prepared, keep_mask, audit = prepare_classification_labels(
            labels, input_date_cutoff_exclusive="2000-03-01"
        )
        np.testing.assert_array_equal(keep_mask, [True, True, False, False])
        self.assertEqual(prepared["source_row"].tolist(), [0, 1])
        self.assertEqual(
            audit["dropped_at_or_after_cutoff_or_invalid_date"], 2
        )

    def test_month_join_and_exact_date_diagnostic_are_separate(self) -> None:
        official = pd.DataFrame(
            {
                "PERMNO": pd.Series([1, 2], dtype="Int64"),
                "Date": pd.to_datetime(["2000-01-31", "2000-02-29"]),
                "month": ["2000-01", "2000-02"],
                "Ret_20d": [0.10, -0.20],
                "Ret_month": [-0.10, -0.20],
                "EWMA_vol": [0.01, 0.02],
                "MarketCap": [100.0, 200.0],
            }
        )
        professor = pd.DataFrame(
            {
                "PERMNO": pd.Series([1, 2, 3], dtype="Int64"),
                "date": pd.to_datetime(["2000-01-28", "2000-02-29", "2000-03-31"]),
                "month": ["2000-01", "2000-02", "2000-03"],
                "EXCHCD": [1.0, 4.0, 1.0],
                "exchange_eligible": [True, False, True],
                "source_observation_number": [1, 1, 1],
                "source_observation_count": [1, 1, 1],
                "is_first_source_observation": [True, True, True],
                "within_first_3_source_observations": [True, True, True],
                "is_last_source_observation": [True, True, True],
            }
        )
        summaries, annual, monthly, unmatched = match_universes(official, professor)
        all_scope = summaries["all"]
        eligible_scope = summaries["exchange_eligible"]
        self.assertEqual(all_scope["matched_keys"], 2)
        self.assertAlmostEqual(all_scope["professor_coverage_rate"], 2 / 3)
        self.assertAlmostEqual(all_scope["exact_date_match_rate_among_matches"], 1 / 2)
        self.assertEqual(all_scope["ret_20d_month_sign_disagreements"], 1)
        self.assertEqual(eligible_scope["professor_keys"], 2)
        self.assertEqual(eligible_scope["matched_keys"], 1)
        self.assertEqual(len(unmatched["all"][0]), 1)
        self.assertEqual(len(annual), 2)
        self.assertEqual(len(monthly), 6)

        exchange = match_by_exchange(official, professor).set_index("EXCHCD")
        self.assertEqual(exchange.loc["1", "professor_keys"], 2)
        self.assertEqual(exchange.loc["4", "matched_keys"], 1)


if __name__ == "__main__":
    unittest.main()
