from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import assign_deciles, ivol_from_moments, next_month_label


class CoreTests(unittest.TestCase):
    def test_next_month_alignment(self) -> None:
        self.assertEqual(next_month_label("1999-12"), "2000-01")
        self.assertEqual(next_month_label("2000-11"), "2000-12")

    def test_deciles_are_deterministic_when_signals_tie(self) -> None:
        frame = pd.DataFrame(
            {
                "PERMNO": list(range(20, 0, -1)),
                "formation_month": ["2000-01"] * 20,
                "ivol_daily": [0.02] * 20,
            }
        )
        deciles = assign_deciles(frame)
        assigned = frame.assign(decile=deciles).sort_values("PERMNO")
        self.assertEqual(assigned.iloc[0]["decile"], 1)
        self.assertEqual(assigned.iloc[-1]["decile"], 10)
        self.assertEqual(assigned["decile"].value_counts().sort_index().tolist(), [2] * 10)

    def test_ivol_matches_direct_residual_standard_deviation(self) -> None:
        rng = np.random.default_rng(7)
        x = np.column_stack([np.ones(25), rng.normal(size=(25, 3))])
        residual_component = rng.normal(scale=0.02, size=25)
        y = x @ np.array([0.001, 0.8, -0.3, 0.2]) + residual_component
        xtx = x.T @ x
        xty = x.T @ y
        ivol, rank = ivol_from_moments(xtx, xty, float(y @ y), len(y))
        beta = np.linalg.solve(xtx, xty)
        expected = np.std(y - x @ beta, ddof=1)
        self.assertEqual(rank, 4)
        self.assertTrue(math.isclose(ivol, expected, rel_tol=1e-10, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
