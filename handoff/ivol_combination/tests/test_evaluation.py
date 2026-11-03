from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate_ivol import build_monthly_portfolios  # noqa: E402
from performance_metrics import (  # noqa: E402
    newey_west_t_stat,
    summarize_return_series,
)
from shared_contract import build_adj_price_outcomes, canonical_month  # noqa: E402


class EvaluationContractTests(unittest.TestCase):
    def professor_prices(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for permno in range(1, 11):
            rows.append(
                {
                    "PERMNO": permno,
                    "date": pd.Timestamp("2020-01-31"),
                    "formation_month": "2020-01",
                    "EXCHCD": 1,
                    "TICKER": f"S{permno}",
                    "COMNAM": f"Stock {permno}",
                    "ADJ_PRC": 100.0,
                }
            )
            rows.append(
                {
                    "PERMNO": permno,
                    "date": pd.Timestamp("2020-02-28"),
                    "formation_month": "2020-02",
                    # An outcome-month exchange change must not remove the return.
                    "EXCHCD": 4 if permno == 1 else 1,
                    "TICKER": f"S{permno}",
                    "COMNAM": f"Stock {permno}",
                    "ADJ_PRC": 100.0 + permno,
                }
            )
        return pd.DataFrame(rows)

    def test_exact_next_month_outcomes_without_survivor_refilter(self) -> None:
        outcomes, audit = build_adj_price_outcomes(
            self.professor_prices(), "2020-01", "2020-01"
        )
        first = outcomes.loc[outcomes["PERMNO"].eq(1)].iloc[0]
        self.assertEqual(first["holding_month"], "2020-02")
        self.assertEqual(first["holding_exchcd"], 4)
        self.assertAlmostEqual(first["next_month_return"], 0.01)
        self.assertFalse(audit["outcome_month_survivor_refilter"])

    def test_equal_weight_deciles_and_spread(self) -> None:
        outcomes, _ = build_adj_price_outcomes(
            self.professor_prices(), "2020-01", "2020-01"
        )
        assignments = outcomes[["PERMNO", "holding_month", "next_month_return"]].copy()
        assignments["ivol_decile"] = range(1, 11)
        monthly, audit = build_monthly_portfolios(assignments)
        q1 = monthly.loc[monthly["portfolio"].eq("Q1"), "return"].iloc[0]
        q10 = monthly.loc[monthly["portfolio"].eq("Q10"), "return"].iloc[0]
        spread = monthly.loc[monthly["portfolio"].eq("Q1-Q10"), "return"].iloc[0]
        self.assertAlmostEqual(q1, 0.01)
        self.assertAlmostEqual(q10, 0.10)
        self.assertAlmostEqual(spread, q1 - q10)
        self.assertEqual(audit["weighting"], "equal_weight")

    def test_performance_convention(self) -> None:
        returns = pd.Series([0.01, 0.03, -0.01, 0.02])
        summary = summarize_return_series(returns, return_kind="total")
        self.assertAlmostEqual(summary["annualized_mean_return"], 12 * returns.mean())
        self.assertAlmostEqual(
            summary["annualized_volatility"], math.sqrt(12) * returns.std(ddof=1)
        )
        self.assertTrue(np.isfinite(newey_west_t_stat(returns, lag=1)))

    def test_true_sharpe_uses_excess_return_volatility(self) -> None:
        returns = pd.Series([0.02, 0.04, 0.03, 0.06])
        risk_free = pd.Series([0.01, 0.02, 0.00, 0.01])
        excess = returns - risk_free
        summary = summarize_return_series(
            returns,
            risk_free=risk_free,
            return_kind="total",
        )
        expected = math.sqrt(12) * excess.mean() / excess.std(ddof=1)
        self.assertAlmostEqual(summary["annualized_sharpe"], expected)

    def test_value_weighting_uses_formation_weights(self) -> None:
        assignments = pd.DataFrame(
            {
                "PERMNO": range(1, 21),
                "holding_month": ["2020-02"] * 20,
                "ivol_decile": np.repeat(range(1, 11), 2),
                "next_month_return": np.tile([0.0, 0.10], 10),
                "formation_market_cap": np.tile([1.0, 3.0], 10),
            }
        )
        monthly, audit = build_monthly_portfolios(
            assignments, weight_column="formation_market_cap"
        )
        q1 = monthly.loc[monthly["portfolio"].eq("Q1"), "return"].iloc[0]
        self.assertAlmostEqual(q1, 0.075)
        self.assertEqual(audit["weighting"], "value_weight")

    def test_compact_month_formats(self) -> None:
        values = pd.Series([202001, 202002.0, "2020-03", "2020-04-30"])
        self.assertEqual(
            canonical_month(values, "month").tolist(),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        )


if __name__ == "__main__":
    unittest.main()
