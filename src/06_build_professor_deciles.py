from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import MONTHLY_FILE, ensure_parent, resolve_data_dir, write_json
from professor_universe import (
    build_professor_assignments,
    load_formation_caps,
    load_professor_stock_data,
    official_formation_universe,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build IVOL deciles and exact next-month ADJ_PRC returns."
    )
    result.add_argument("--data-dir")
    result.add_argument("--professor-csv", required=True)
    result.add_argument("--ivol", required=True)
    result.add_argument("--formation-start", required=True, help="YYYY-MM")
    result.add_argument("--formation-end", required=True, help="YYYY-MM")
    result.add_argument("--output", required=True)
    result.add_argument("--log", required=True)
    result.add_argument("--chunksize", type=int, default=300_000)
    return result


def main() -> None:
    args = parser().parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    professor, professor_audit = load_professor_stock_data(args.professor_csv)
    formation = official_formation_universe(
        professor,
        formation_start=args.formation_start,
        formation_end=args.formation_end,
    )
    caps, cap_audit = load_formation_caps(
        data_dir / MONTHLY_FILE,
        formation[["PERMNO", "formation_month"]],
        chunksize=args.chunksize,
    )
    ivol = pd.read_csv(args.ivol, compression="gzip", low_memory=False)
    assignments, assignment_audit = build_professor_assignments(
        professor,
        ivol,
        caps,
        formation_start=args.formation_start,
        formation_end=args.formation_end,
    )

    output = ensure_parent(args.output)
    assignments.to_csv(output, index=False, compression="gzip")
    audit = {
        "professor_source": professor_audit,
        "market_cap_source": {
            "file": Path(data_dir / MONTHLY_FILE).name,
            "purpose": "Optional value weighting only; not used to define membership or returns",
            **cap_audit,
        },
        **assignment_audit,
    }
    write_json(audit, args.log)
    print(f"Wrote {len(assignments):,} professor-universe decile assignments to {output}")


if __name__ == "__main__":
    main()
