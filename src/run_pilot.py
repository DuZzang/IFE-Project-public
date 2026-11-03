from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import resolve_data_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run Phase 0 and the 2000 IVOL pilot.")
    result.add_argument("--data-dir")
    result.add_argument("--skip-inventory", action="store_true")
    result.add_argument("--chunksize", type=int, default=500_000)
    return result


def run(command: list[str], root: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    args = parser().parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    src = root / "src"
    derived = root / "derived" / "pilot_2000"
    outputs = root / "outputs" / "pilot_2000"
    logs = root / "logs"

    if not args.skip_inventory:
        run(
            [
                python,
                str(src / "00_inventory.py"),
                "--data-dir",
                str(data_dir),
                "--output-json",
                str(root / "reports" / "phase0_inventory.json"),
                "--output-md",
                str(root / "reports" / "phase0_inventory.md"),
                "--chunksize",
                str(args.chunksize),
            ],
            root,
        )

    daily_panel = derived / "daily_panel_1999-12_2000-11.csv.gz"
    ivol = derived / "ivol_1999-12_2000-11.csv.gz"
    assignments = derived / "ivol_deciles_returns_2000.csv.gz"

    run(
        [
            python,
            str(src / "01_build_daily_panel.py"),
            "--data-dir",
            str(data_dir),
            "--formation-start",
            "1999-12",
            "--formation-end",
            "2000-11",
            "--output",
            str(daily_panel),
            "--log",
            str(logs / "01_build_daily_panel_pilot.json"),
            "--chunksize",
            str(args.chunksize),
        ],
        root,
    )
    run(
        [
            python,
            str(src / "02_compute_ivol.py"),
            "--input",
            str(daily_panel),
            "--output",
            str(ivol),
            "--log",
            str(logs / "02_compute_ivol_pilot.json"),
            "--min-obs",
            "15",
            "--chunksize",
            str(args.chunksize),
        ],
        root,
    )
    run(
        [
            python,
            str(src / "03_build_deciles.py"),
            "--data-dir",
            str(data_dir),
            "--ivol",
            str(ivol),
            "--output",
            str(assignments),
            "--log",
            str(logs / "03_build_deciles_pilot.json"),
            "--chunksize",
            str(args.chunksize),
        ],
        root,
    )
    run(
        [
            python,
            str(src / "04_evaluate.py"),
            "--input",
            str(assignments),
            "--output-dir",
            str(outputs),
            "--log",
            str(logs / "04_evaluate_pilot.json"),
        ],
        root,
    )
    print("Pilot complete.")


if __name__ == "__main__":
    main()
