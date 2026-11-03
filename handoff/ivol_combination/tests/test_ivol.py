from __future__ import annotations

import math
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from build_combination_signal import build_signal  # noqa: E402
from compute_ivol import compute_ivol_from_moments  # noqa: E402
from ivol_core import ivol_from_moments  # noqa: E402
from merge_signal import merge_signal  # noqa: E402
from validate_signal import validate  # noqa: E402


def professor_toy(n_stocks: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PERMNO": range(1, n_stocks + 1),
            "date": [pd.Timestamp("2020-01-31")] * n_stocks,
            "formation_month": ["2020-01"] * n_stocks,
            "EXCHCD": [1] * n_stocks,
            "TICKER": [f"S{i}" for i in range(1, n_stocks + 1)],
            "COMNAM": [f"Stock {i}" for i in range(1, n_stocks + 1)],
        }
    )


def ivol_toy(n_stocks: int = 10) -> pd.DataFrame:
    daily = np.arange(1, n_stocks + 1, dtype=float) / 1_000.0
    return pd.DataFrame(
        {
            "PERMNO": range(1, n_stocks + 1),
            "formation_month": ["2020-01"] * n_stocks,
            "ivol_daily": daily,
            "ivol_annualized": daily * math.sqrt(252.0),
            "n_daily_obs": [20] * n_stocks,
            "regression_rank": [4] * n_stocks,
        }
    )


class IvolHandoffTests(unittest.TestCase):
    def test_moment_ivol_matches_direct_ols_residual_std(self) -> None:
        rng = np.random.default_rng(7)
        x = np.column_stack([np.ones(25), rng.normal(size=(25, 3))])
        y = x @ np.array([0.001, 0.8, -0.3, 0.2]) + rng.normal(
            scale=0.02, size=25
        )
        xtx = x.T @ x
        xty = x.T @ y
        ivol, rank = ivol_from_moments(xtx, xty, float(y @ y), len(y))
        beta = np.linalg.solve(xtx, xty)
        expected = np.std(y - x @ beta, ddof=1)
        self.assertEqual(rank, 4)
        self.assertTrue(math.isclose(ivol, expected, rel_tol=1e-10, abs_tol=1e-12))

    def test_insufficient_and_singular_stock_months_are_rejected(self) -> None:
        moments = pd.DataFrame(
            [
                {
                    "PERMNO": 1,
                    "formation_month": "2020-01",
                    "n": 14,
                    "y": 0.0,
                    "m": 0.0,
                    "s": 0.0,
                    "h": 0.0,
                    "mm": 1.0,
                    "ms": 0.0,
                    "mh": 0.0,
                    "ss": 1.0,
                    "sh": 0.0,
                    "hh": 1.0,
                    "my": 0.0,
                    "sy": 0.0,
                    "hy": 0.0,
                    "yy": 1.0,
                },
                {
                    "PERMNO": 2,
                    "formation_month": "2020-01",
                    "n": 20,
                    "y": 0.0,
                    "m": 0.0,
                    "s": 0.0,
                    "h": 0.0,
                    "mm": 0.0,
                    "ms": 0.0,
                    "mh": 0.0,
                    "ss": 0.0,
                    "sh": 0.0,
                    "hh": 0.0,
                    "my": 0.0,
                    "sy": 0.0,
                    "hy": 0.0,
                    "yy": 1.0,
                },
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "No valid"):
            compute_ivol_from_moments(moments, min_obs=15)

    def test_cli_excludes_nonavailable_crsp_return_flag(self) -> None:
        rng = np.random.default_rng(17)
        factors = rng.normal(scale=0.01, size=(16, 3))
        x = np.column_stack([np.ones(16), factors])
        y = x @ np.array([0.001, 0.7, -0.2, 0.3]) + rng.normal(
            scale=0.005, size=16
        )
        y[-1] = 2.0
        panel = pd.DataFrame(
            {
                "PERMNO": [1] * 16,
                "formation_month": ["2020-01"] * 16,
                "excess_return": y,
                "mkt_rf": factors[:, 0],
                "smb": factors[:, 1],
                "hml": factors[:, 2],
                "DlyRetMissFlg": ["NA"] * 15 + ["MV"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "daily.csv"
            output = root / "ivol.csv"
            log = root / "log.json"
            panel.to_csv(source, index=False)
            subprocess.run(
                [
                    sys.executable,
                    str(SRC / "compute_ivol.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--log",
                    str(log),
                    "--min-obs",
                    "15",
                    "--chunksize",
                    "7",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = pd.read_csv(output)
            audit = json.loads(log.read_text(encoding="utf-8"))
        x_valid = x[:15]
        y_valid = y[:15]
        beta = np.linalg.lstsq(x_valid, y_valid, rcond=None)[0]
        expected = np.std(y_valid - x_valid @ beta, ddof=1)
        self.assertAlmostEqual(result.iloc[0]["ivol_daily"], expected, places=12)
        self.assertEqual(result.iloc[0]["n_daily_obs"], 15)
        self.assertEqual(audit["rows_excluded_nonavailable_return_flag"], 1)

    def test_cli_requires_return_availability_flag_by_default(self) -> None:
        index = np.arange(20)
        frame = pd.DataFrame(
            {
                "PERMNO": [1] * 20,
                "formation_month": ["2020-01"] * 20,
                "excess_return": 0.002 + 0.01 * np.sin(index / 2) + index / 100_000,
                "mkt_rf": np.linspace(-0.01, 0.01, 20),
                "smb": np.sin(index) / 100,
                "hml": np.cos(index) / 100,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "panel.csv"
            frame.to_csv(source, index=False)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "compute_ivol.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(root / "ivol.csv"),
                    "--log",
                    str(root / "log.json"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            allowed = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "compute_ivol.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(root / "ivol_prevalidated.csv"),
                    "--log",
                    str(root / "prevalidated_log.json"),
                    "--allow-prevalidated-without-return-flag",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("missing DlyRetMissFlg", completed.stderr)
        self.assertEqual(allowed.returncode, 0, msg=allowed.stderr)


    def test_combination_signal_direction_and_timing(self) -> None:
        signal, audit = build_signal(
            professor_toy(),
            ivol_toy(),
            formation_start="2020-01",
            formation_end="2020-01",
        )
        low = signal.loc[signal["PERMNO"].eq(1)].iloc[0]
        high = signal.loc[signal["PERMNO"].eq(10)].iloc[0]
        self.assertEqual(low["holding_month"], "2020-02")
        self.assertEqual(low["ivol_decile"], 1)
        self.assertEqual(high["ivol_decile"], 10)
        self.assertAlmostEqual(low["ivol_rank_pct"], 0.0)
        self.assertAlmostEqual(high["ivol_rank_pct"], 1.0)
        self.assertAlmostEqual(low["low_ivol_score"], 1.0)
        self.assertAlmostEqual(high["low_ivol_score"], -1.0)
        self.assertTrue(pd.api.types.is_integer_dtype(signal["n_daily_obs"]))
        self.assertTrue(pd.api.types.is_integer_dtype(signal["regression_rank"]))
        self.assertTrue(pd.api.types.is_integer_dtype(signal["ivol_decile"]))
        self.assertEqual(audit["formation_rows_without_ivol"], 0)
        self.assertEqual(validate(signal)["failed_checks"], "")

    def test_missing_ivol_is_not_imputed(self) -> None:
        signal, audit = build_signal(
            professor_toy(),
            ivol_toy().iloc[:-1].copy(),
            formation_start="2020-01",
            formation_end="2020-01",
        )
        self.assertEqual(len(signal), 9)
        self.assertEqual(audit["formation_rows_without_ivol"], 1)
        self.assertNotIn(10, signal["PERMNO"].tolist())

    def test_outside_professor_key_is_rejected(self) -> None:
        outside = ivol_toy()
        outside.loc[0, "PERMNO"] = 999
        with self.assertRaisesRegex(ValueError, "outside the professor universe"):
            build_signal(
                professor_toy(),
                outside,
                formation_start="2020-01",
                formation_end="2020-01",
            )

    def test_merge_can_align_on_holding_month_without_lookahead(self) -> None:
        signal, _ = build_signal(
            professor_toy(),
            ivol_toy(),
            formation_start="2020-01",
            formation_end="2020-01",
        )
        panel = pd.DataFrame(
            {
                "PERMNO": range(1, 11),
                "portfolio_month": ["2020-02"] * 10,
                "other_factor": np.linspace(-1, 1, 10),
            }
        )
        merged, audit = merge_signal(
            panel,
            signal,
            id_column="PERMNO",
            panel_month_column="portfolio_month",
            signal_month_column="holding_month",
        )
        self.assertEqual(len(merged), 10)
        self.assertEqual(audit["match_rate"], 1.0)
        self.assertEqual(merged["ivol_daily"].isna().sum(), 0)

    def test_merge_rejects_existing_ivol_output_columns(self) -> None:
        signal, _ = build_signal(
            professor_toy(),
            ivol_toy(),
            formation_start="2020-01",
            formation_end="2020-01",
        )
        panel = pd.DataFrame(
            {
                "PERMNO": range(1, 11),
                "formation_month": ["2020-01"] * 10,
                "ivol_daily": np.nan,
            }
        )
        with self.assertRaisesRegex(ValueError, "already contains IVOL output"):
            merge_signal(
                panel,
                signal,
                id_column="PERMNO",
                panel_month_column="formation_month",
                signal_month_column="formation_month",
            )

    def test_merge_accepts_compact_numeric_yyyymm(self) -> None:
        signal, _ = build_signal(
            professor_toy(),
            ivol_toy(),
            formation_start="2020-01",
            formation_end="2020-01",
        )
        panel = pd.DataFrame(
            {
                "PERMNO": range(1, 11),
                "panel_month": [202001] * 10,
            }
        )
        merged, audit = merge_signal(
            panel,
            signal,
            id_column="PERMNO",
            panel_month_column="panel_month",
            signal_month_column="formation_month",
        )
        self.assertEqual(audit["match_rate"], 1.0)
        self.assertEqual(merged["ivol_daily"].isna().sum(), 0)

    def test_validator_rejects_nonfinite_or_reversed_signal(self) -> None:
        signal, _ = build_signal(
            professor_toy(),
            ivol_toy(),
            formation_start="2020-01",
            formation_end="2020-01",
        )
        nonfinite = signal.copy()
        nonfinite.loc[0, "ivol_daily"] = np.inf
        with self.assertRaisesRegex(ValueError, "validation failed"):
            validate(nonfinite)

        reversed_score = signal.copy()
        reversed_score["low_ivol_score"] *= -1.0
        with self.assertRaisesRegex(ValueError, "validation failed"):
            validate(reversed_score)

        reversed_deciles = signal.copy()
        reversed_deciles["ivol_decile"] = 11 - reversed_deciles["ivol_decile"]
        with self.assertRaisesRegex(ValueError, "validation failed"):
            validate(reversed_deciles)

        missing_decile = signal.loc[signal["ivol_decile"].ne(10)].copy()
        with self.assertRaisesRegex(ValueError, "validation failed"):
            validate(missing_decile)

        leaked = signal.assign(next_month_ret=0.01)
        with self.assertRaisesRegex(ValueError, "validation failed"):
            validate(leaked)


if __name__ == "__main__":
    unittest.main()
