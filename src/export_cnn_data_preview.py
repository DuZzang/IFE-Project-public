from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a human-readable preview of the official I20 CNN data."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--image-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2001)
    parser.add_argument("--label-sample-size", type=int, default=200)
    parser.add_argument("--image-sample-size", type=int, default=12)
    return parser.parse_args()


def json_number(value: object) -> int | float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return float(value)


def main() -> None:
    args = parse_args()
    src_dir = args.project_root / "src"
    sys.path.insert(0, str(src_dir))

    from cnn_official_data import (  # pylint: disable=import-outside-toplevel
        IMAGE_SHAPE,
        catalog_archive,
        iter_year_image_batches,
        read_year_labels,
        validate_label_partition,
        verify_official_archive_hash,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observed_sha256, hash_match = verify_official_archive_hash(args.image_zip)
    catalog = catalog_archive(args.image_zip)
    partition = next((item for item in catalog if item.year == args.year), None)
    if partition is None:
        raise KeyError(f"Year {args.year} is not present in the archive")

    raw_labels = read_year_labels(
        args.image_zip, args.year, catalog=catalog
    )
    profile, labels = validate_label_partition(partition, raw_labels)
    row_count = len(labels)

    label_count = min(max(args.label_sample_size, 1), row_count)
    image_count = min(max(args.image_sample_size, 1), row_count)
    image_indices = np.unique(
        np.linspace(0, row_count - 1, image_count, dtype=np.int64)
    )
    label_indices = np.unique(
        np.concatenate(
            [
                np.linspace(0, row_count - 1, label_count, dtype=np.int64),
                image_indices,
            ]
        )
    )
    image_index_set = set(int(item) for item in image_indices)

    selected = labels.iloc[label_indices].copy()
    selected.insert(0, "source_row", label_indices.astype(np.int64))
    selected["image_in_preview"] = selected["source_row"].isin(image_index_set)
    selected["Date"] = selected["Date"].dt.strftime("%Y-%m-%d")

    export_columns = [
        "source_row",
        "Date",
        "PERMNO",
        "MarketCap",
        "Ret_5d",
        "Ret_20d",
        "Ret_60d",
        "Ret_month",
        "EWMA_vol",
        "image_in_preview",
    ]
    selected = selected[export_columns]
    rows: list[dict[str, object]] = []
    for record in selected.to_dict(orient="records"):
        rows.append(
            {
                "source_row": int(record["source_row"]),
                "Date": str(record["Date"]),
                "PERMNO": int(record["PERMNO"]),
                "MarketCap": json_number(record["MarketCap"]),
                "Ret_5d": json_number(record["Ret_5d"]),
                "Ret_20d": json_number(record["Ret_20d"]),
                "Ret_60d": json_number(record["Ret_60d"]),
                "Ret_month": json_number(record["Ret_month"]),
                "EWMA_vol": json_number(record["EWMA_vol"]),
                "image_in_preview": bool(record["image_in_preview"]),
            }
        )

    selected_images: dict[int, np.ndarray] = {}
    final_target = int(image_indices[-1])
    for offset, batch in iter_year_image_batches(
        args.image_zip,
        args.year,
        batch_size=512,
        max_images=final_target + 1,
        catalog=catalog,
    ):
        stop = offset + len(batch)
        candidates = image_indices[(image_indices >= offset) & (image_indices < stop)]
        for row_number in candidates:
            selected_images[int(row_number)] = np.array(
                batch[int(row_number) - offset], copy=True
            )

    missing_images = sorted(image_index_set.difference(selected_images))
    if missing_images:
        raise RuntimeError(f"Failed to extract preview rows: {missing_images}")

    columns = 4
    rows_n = int(np.ceil(len(image_indices) / columns))
    fig, axes = plt.subplots(rows_n, columns, figsize=(13.5, 10.0), dpi=170)
    axes_flat = np.asarray(axes).reshape(-1)
    for axis, row_number in zip(axes_flat, image_indices, strict=False):
        label = labels.iloc[int(row_number)]
        axis.imshow(
            selected_images[int(row_number)],
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        ret_month = pd.to_numeric(label["Ret_month"], errors="coerce")
        ret_text = "NA" if pd.isna(ret_month) else f"{float(ret_month):+.3f}"
        axis.set_title(
            f"row {int(row_number):,} | PERMNO {int(label['PERMNO'])}\n"
            f"{pd.Timestamp(label['Date']).date()} | Ret_month {ret_text}",
            fontsize=8.2,
            pad=5,
        )
        axis.axis("off")
    for axis in axes_flat[len(image_indices) :]:
        axis.axis("off")
    fig.suptitle(
        f"Author-Released I20 Chart Images — {args.year} Sample",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        f"Each 64×60 image is aligned to the label row shown above it. "
        f"Source: img_data.zip ({IMAGE_SHAPE[0]}×{IMAGE_SHAPE[1]} uint8).",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.02, 0.035, 0.98, 0.97), w_pad=1.2, h_pad=1.4)
    image_path = args.output_dir / "author_i20_image_preview.png"
    fig.savefig(image_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    payload = {
        "source": {
            "archive_filename": args.image_zip.name,
            "archive_sha256": observed_sha256,
            "official_hash_match": bool(hash_match),
            "partition_year": args.year,
            "image_member": partition.image_member,
            "label_member": partition.label_member,
        },
        "profile": profile,
        "sample": {
            "label_rows": len(rows),
            "image_rows": len(image_indices),
            "image_indices": [int(item) for item in image_indices],
        },
        "rows": rows,
    }
    json_path = args.output_dir / "author_i20_preview_data.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "json": str(json_path),
                "image": str(image_path),
                "partition_rows": row_count,
                "label_sample_rows": len(rows),
                "image_sample_rows": len(image_indices),
                "date_min": profile["min_date"],
                "date_max": profile["max_date"],
                "hash_match": hash_match,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
