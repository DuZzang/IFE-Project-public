from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
SPEC = importlib.util.spec_from_file_location("evaluate", SRC / "04_evaluate.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load 04_evaluate.py for testing.")
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


def summary(scale: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portfolio": [*(f"Q{i}" for i in range(1, 11)), "Q1-Q10"],
            "annualized_mean_return": [
                *(scale * value for value in np.linspace(0.10, -0.10, 10)),
                scale * 0.20,
            ],
            "annualized_volatility": [
                *(scale * value for value in np.linspace(0.15, 0.30, 10)),
                scale * 0.25,
            ],
        }
    )


class EvaluateFigureTests(unittest.TestCase):
    def test_build_decile_figure_data_has_exact_decile_grain(self) -> None:
        chart = EVALUATE.build_decile_figure_data(summary(1.0), summary(2.0))
        self.assertEqual(chart["decile"].tolist(), list(range(1, 11)))
        self.assertEqual(len(chart), 10)
        self.assertAlmostEqual(chart.iloc[0]["equal_weight_annualized_mean_return"], 0.10)
        self.assertAlmostEqual(chart.iloc[0]["value_weight_annualized_mean_return"], 0.20)

    def test_two_panel_figure_writes_png(self) -> None:
        chart = EVALUATE.build_decile_figure_data(summary(1.0), summary(2.0))
        with tempfile.TemporaryDirectory() as directory:
            png = Path(directory) / "figure.png"
            EVALUATE.save_two_panel_decile_figure(
                chart,
                png,
                "2000-12",
                "2020-11",
                "2001-01",
                "2020-12",
            )
            self.assertGreater(png.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
