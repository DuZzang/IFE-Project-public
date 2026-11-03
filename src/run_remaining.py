from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import resolve_data_dir


FORMATION_START = "2000-12"
FORMATION_END = "2020-11"
RETURN_START = "2001-01"
RETURN_END = "2020-12"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run the remaining 2001-2020 IVOL sample after the 2000 pilot."
    )
    result.add_argument("--data-dir")
    result.add_argument("--chunksize", type=int, default=500_000)
    result.add_argument("--min-obs", type=int, default=15)
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
    derived = root / "derived" / "remaining_2001_2020"
    outputs = root / "outputs" / "remaining_2001_2020"
    logs = root / "logs"

    daily_panel = derived / "daily_panel_2000-12_2020-11.csv.gz"
    ivol = derived / "ivol_2000-12_2020-11.csv.gz"
    assignments = derived / "ivol_deciles_returns_2001-2020.csv.gz"

    run(
        [
            python,
            str(src / "01_build_daily_panel.py"),
            "--data-dir",
            str(data_dir),
            "--formation-start",
            FORMATION_START,
            "--formation-end",
            FORMATION_END,
            "--output",
            str(daily_panel),
            "--log",
            str(logs / "01_build_daily_panel_remaining_2001_2020.json"),
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
            str(logs / "02_compute_ivol_remaining_2001_2020.json"),
            "--min-obs",
            str(args.min_obs),
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
            str(logs / "03_build_deciles_remaining_2001_2020.json"),
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
            str(logs / "04_evaluate_remaining_2001_2020.json"),
        ],
        root,
    )
    print(
        f"Remaining sample complete: formation {FORMATION_START} to {FORMATION_END}; "
        f"returns {RETURN_START} to {RETURN_END}."
    )


if __name__ == "__main__":
    main()
