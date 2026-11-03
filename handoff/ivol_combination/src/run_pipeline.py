from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ivol_core import resolve_data_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Rebuild the professor-universe IVOL signal for factor combination."
    )
    result.add_argument("--data-dir")
    result.add_argument("--professor-csv", required=True)
    result.add_argument("--formation-start", required=True, help="YYYY-MM")
    result.add_argument("--formation-end", required=True, help="YYYY-MM")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--chunksize", type=int, default=500_000)
    result.add_argument("--min-obs", type=int, default=15)
    return result


def run(command: list[str], root: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    args = parser().parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    professor_csv = Path(args.professor_csv).expanduser().resolve()
    if not professor_csv.is_file():
        raise FileNotFoundError(f"Professor CSV does not exist: {professor_csv}")
    if args.chunksize <= 0:
        raise ValueError("chunksize must be positive")
    # The shared signal contract fixes the baseline at 15 observations.  Allowing
    # a lower value here would only fail later when build_combination_signal
    # validates the IVOL file, so reject it at the entry point with a clear error.
    if args.min_obs < 15:
        raise ValueError("min-obs must be at least 15 for the shared IVOL signal")

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    output_dir = Path(args.output_dir).expanduser().resolve()
    intermediate = output_dir / "intermediate"
    logs = output_dir / "logs"
    intermediate.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    daily_panel = intermediate / (
        f"daily_panel_{args.formation_start}_{args.formation_end}.csv.gz"
    )
    ivol = output_dir / f"ivol_{args.formation_start}_{args.formation_end}.csv.gz"
    signal = output_dir / (
        f"ivol_combination_signal_{args.formation_start}_{args.formation_end}.csv.gz"
    )
    expected_months = len(
        pd.period_range(args.formation_start, args.formation_end, freq="M")
    )

    run(
        [
            sys.executable,
            str(src / "build_daily_panel.py"),
            "--data-dir",
            str(data_dir),
            "--professor-csv",
            str(professor_csv),
            "--formation-start",
            args.formation_start,
            "--formation-end",
            args.formation_end,
            "--output",
            str(daily_panel),
            "--log",
            str(logs / "build_daily_panel.json"),
            "--chunksize",
            str(args.chunksize),
        ],
        root,
    )
    run(
        [
            sys.executable,
            str(src / "compute_ivol.py"),
            "--input",
            str(daily_panel),
            "--output",
            str(ivol),
            "--log",
            str(logs / "compute_ivol.json"),
            "--min-obs",
            str(args.min_obs),
            "--chunksize",
            str(args.chunksize),
        ],
        root,
    )
    run(
        [
            sys.executable,
            str(src / "build_combination_signal.py"),
            "--professor-csv",
            str(professor_csv),
            "--ivol",
            str(ivol),
            "--formation-start",
            args.formation_start,
            "--formation-end",
            args.formation_end,
            "--output",
            str(signal),
            "--log",
            str(logs / "build_combination_signal.json"),
        ],
        root,
    )
    run(
        [
            sys.executable,
            str(src / "validate_signal.py"),
            "--input",
            str(signal),
            "--log",
            str(logs / "validate_signal.json"),
            "--expected-months",
            str(expected_months),
        ],
        root,
    )
    print("IVOL combination handoff complete")
    print(f"Raw monthly IVOL: {ivol}")
    print(f"Combination signal: {signal}")


if __name__ == "__main__":
    main()
