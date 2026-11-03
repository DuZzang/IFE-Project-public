from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from professor_universe import build_professor_assignments  # noqa: E402


EVAL_SPEC = importlib.util.spec_from_file_location(
    "evaluate_professor", SRC / "07_evaluate_professor.py"
)
if EVAL_SPEC is None or EVAL_SPEC.loader is None:
    raise RuntimeError("Could not load 07_evaluate_professor.py for testing.")
EVALUATE = importlib.util.module_from_spec(EVAL_SPEC)
EVAL_SPEC.loader.exec_module(EVALUATE)


def professor_toy() -> pd.DataFrame:
    rows = []
    for permno in range(1, 11):
        rows.append(
            {
                "PERMNO": permno,
                "date": pd.Timestamp("2020-01-31"),
                "formation_month": "2020-01",
                "EXCHCD": 1,
                "TICKER": f"S{permno}",
                "COMNAM": f"Stock {permno}",
                "ADJ_PRC": 10.0,
            }
        )
        if permno < 10:
            rows.append(
                {
                    "PERMNO": permno,
                    "date": pd.Timestamp("2020-02-28"),
                    "formation_month": "2020-02",
                    "EXCHCD": 4 if permno == 1 else 1,
                    "TICKER": f"S{permno}",
                    "COMNAM": f"Stock {permno}",
                    "ADJ_PRC": 11.0,
                }
            )
    rows.append(
        {
            "PERMNO": 10,
            "date": pd.Timestamp("2020-03-31"),
            "formation_month": "2020-03",
            "EXCHCD": 1,
            "TICKER": "S10",
            "COMNAM": "Stock 10",
            "ADJ_PRC": 20.0,
        }
    )
    return pd.DataFrame(rows)


def ivol_toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PERMNO": range(1, 11),
            "formation_month": ["2020-01"] * 10,
            "ivol_daily": np.arange(1, 11, dtype=float) / 100.0,
            "ivol_annualized": np.arange(1, 11, dtype=float) / 100.0 * math.sqrt(252),
            "n_daily_obs": [20] * 10,
            "regression_rank": [4] * 10,
        }
    )


def caps_toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PERMNO": range(1, 11),
            "formation_month": ["2020-01"] * 10,
            "formation_market_cap": np.arange(101, 111, dtype=float),
        }
    )


class ProfessorUniverseTests(unittest.TestCase):
    def test_exact_next_month_join_and_no_outcome_exchange_refilter(self) -> None:
        assignments, audit = build_professor_assignments(
            professor_toy(),
            ivol_toy(),
            caps_toy(),
            formation_start="2020-01",
            formation_end="2020-01",
        )
        first = assignments.loc[assignments["PERMNO"].eq(1)].iloc[0]
        skipped = assignments.loc[assignments["PERMNO"].eq(10)].iloc[0]

        self.assertAlmostEqual(first["next_month_return"], 0.10)
        self.assertEqual(first["next_month_exchcd"], 4)
        self.assertTrue(pd.isna(skipped["next_month_return"]))
        self.assertEqual(skipped["next_month"], "2020-02")
        self.assertEqual(skipped["decile"], 10)
        self.assertEqual(audit["rows_missing_exact_next_month_price_return"], 1)

    def test_ivol_key_outside_professor_universe_is_rejected(self) -> None:
        outside = pd.concat(
            [
                ivol_toy(),
                pd.DataFrame(
                    {
                        "PERMNO": [999],
                        "formation_month": ["2020-01"],
                        "ivol_daily": [0.5],
                        "ivol_annualized": [0.5 * math.sqrt(252)],
                        "n_daily_obs": [20],
                        "regression_rank": [4],
                    }
                ),
            ],
            ignore_index=True,
        )
        with self.assertRaises(ValueError):
            build_professor_assignments(
                professor_toy(), outside, caps_toy(), "2020-01", "2020-01"
            )

    def test_annualization_uses_monthly_portfolio_time_series(self) -> None:
        rows = []
        for month_index, month in enumerate(["2020-02", "2020-03", "2020-04"]):
            for decile in range(1, 11):
                rows.append(
                    {
                        "PERMNO": 100 * month_index + decile,
                        "formation_month": str(pd.Period(month, freq="M") - 1),
                        "next_month": month,
                        "decile": decile,
                        "next_month_return": 0.01 * month_index + 0.001 * decile,
                        "formation_market_cap": 1.0,
                    }
                )
        assignments = pd.DataFrame(rows)
        equal, value, details = EVALUATE.build_monthly_portfolios(assignments)
        equal_summary = EVALUATE.portfolio_summary(
            equal, details["equal_count_means"]
        )
        q1 = equal_summary.loc[equal_summary["portfolio"].eq("Q1")].iloc[0]
        expected = pd.Series([0.001, 0.011, 0.021])
        self.assertAlmostEqual(q1["annualized_mean_return"], 12 * expected.mean())
        self.assertAlmostEqual(
            q1["annualized_volatility"], math.sqrt(12) * expected.std(ddof=1)
        )
        self.assertEqual(details["expected_return_months"], 3)
        self.assertEqual(len(value), 33)

    def test_professor_figure_renders_at_expected_size(self) -> None:
        base = pd.DataFrame(
            {
                "portfolio": [*(f"Q{i}" for i in range(1, 11)), "Q1-Q10"],
                "n_months": [251] * 11,
                "annualized_mean_return": np.linspace(0.12, 0.03, 11),
                "annualized_volatility": np.linspace(0.12, 0.36, 11),
                "average_constituents_with_return": [400.0] * 11,
            }
        )
        chart = EVALUATE.build_figure_data(base, base)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure.png"
            EVALUATE.save_two_panel_figure(
                chart,
                path,
                "2000-01",
                "2020-11",
                "2000-02",
                "2020-12",
                251,
            )
            image = plt.imread(path)
            self.assertEqual(image.shape[1], 3600)
            self.assertEqual(image.shape[0], 2040)


if __name__ == "__main__":
    unittest.main()
