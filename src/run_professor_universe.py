from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import resolve_data_dir


FORMATION_START = "2000-01"
FORMATION_END = "2020-11"
RETURN_START = "2000-02"
RETURN_END = "2020-12"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run the official IVOL analysis on the professor CSV universe."
    )
    result.add_argument("--data-dir")
    result.add_argument("--professor-csv", required=True)
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

    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    src = root / "src"
    derived = root / "derived" / "professor_universe_2000_2020"
    outputs = root / "outputs" / "professor_universe_2000_2020"
    logs = root / "logs"

    daily_panel = derived / "daily_panel_2000-01_2020-11.csv.gz"
    ivol = derived / "ivol_2000-01_2020-11.csv.gz"
    assignments = derived / "ivol_deciles_adj_prc_returns_2000-2020.csv.gz"
    report_figure = (
        root
        / "reports"
        / "figures"
        / "ivol_decile_return_volatility_professor_universe_2000_2020.png"
    )

    run(
        [
            python,
            str(src / "05_build_professor_daily_panel.py"),
            "--data-dir",
            str(data_dir),
            "--professor-csv",
            str(professor_csv),
            "--formation-start",
            FORMATION_START,
            "--formation-end",
            FORMATION_END,
            "--output",
            str(daily_panel),
            "--log",
            str(logs / "05_build_professor_daily_panel_2000_2020.json"),
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
            str(logs / "06_compute_ivol_professor_universe_2000_2020.json"),
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
            str(src / "06_build_professor_deciles.py"),
            "--data-dir",
            str(data_dir),
            "--professor-csv",
            str(professor_csv),
            "--ivol",
            str(ivol),
            "--formation-start",
            FORMATION_START,
            "--formation-end",
            FORMATION_END,
            "--output",
            str(assignments),
            "--log",
            str(logs / "07_build_professor_deciles_2000_2020.json"),
            "--chunksize",
            str(args.chunksize),
        ],
        root,
    )
    run(
        [
            python,
            str(src / "07_evaluate_professor.py"),
            "--input",
            str(assignments),
            "--output-dir",
            str(outputs),
            "--log",
            str(logs / "08_evaluate_professor_universe_2000_2020.json"),
            "--report-figure",
            str(report_figure),
        ],
        root,
    )
    run(
        [
            python,
            str(src / "08_validate_professor_run.py"),
            "--assignments",
            str(assignments),
            "--output-dir",
            str(outputs),
            "--report-figure",
            str(report_figure),
            "--professor-csv",
            str(professor_csv),
            "--log",
            str(logs / "09_validate_professor_universe_2000_2020.json"),
        ],
        root,
    )
    print(
        "Professor-universe run complete: "
        f"formation {FORMATION_START} to {FORMATION_END}; "
        f"returns {RETURN_START} to {RETURN_END}."
    )


if __name__ == "__main__":
    main()
