from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_cnn_oos_deciles import (  # noqa: E402
    assign_cnn_deciles,
    build_monthly_portfolios,
    ensemble_up_probabilities,
)


class _FixedLogit(torch.nn.Module):
    def __init__(self, logits: tuple[float, float]) -> None:
        super().__init__()
        self.register_buffer("fixed", torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fixed.repeat(len(inputs), 1)


class CnnOosDecileTests(unittest.TestCase):
    def test_assign_cnn_deciles_is_balanced_and_ordered(self) -> None:
        frame = pd.DataFrame(
            {
                "PERMNO": np.arange(10000, 10023),
                "formation_month": ["2001-01"] * 23,
                "cnn_prob_up": np.linspace(0.1, 0.9, 23),
            }
        )
        frame["decile"] = assign_cnn_deciles(frame)
        counts = frame.groupby("decile").size()
        self.assertEqual(
            sorted(counts.index.astype(int).tolist()), list(range(1, 11))
        )
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        means = frame.groupby("decile")["cnn_prob_up"].mean()
        self.assertTrue(means.is_monotonic_increasing)

    def test_assign_cnn_deciles_breaks_ties_by_permno(self) -> None:
        frame = pd.DataFrame(
            {
                "PERMNO": list(range(20, 0, -1)),
                "formation_month": ["2001-01"] * 20,
                "cnn_prob_up": [0.5] * 20,
                "future_return": np.linspace(1.0, -1.0, 20),
            }
        )
        frame["decile"] = assign_cnn_deciles(frame)
        ordered = frame.sort_values("PERMNO")
        self.assertEqual(
            ordered["decile"].tolist(),
            [value for value in range(1, 11) for _ in range(2)],
        )

    def test_ensemble_probability_is_mean_of_model_probabilities(self) -> None:
        inputs = torch.zeros((3, 1, 64, 60), dtype=torch.float32)
        models = [_FixedLogit((0.0, 0.0)), _FixedLogit((0.0, 2.0))]
        result = ensemble_up_probabilities(models, inputs)
        expected = (
            0.5 + float(torch.softmax(torch.tensor([0.0, 2.0]), dim=0)[1])
        ) / 2
        self.assertTrue(torch.allclose(result, torch.full((3,), expected)))

    def test_long_short_direction_is_high_score_minus_low_score(self) -> None:
        rows = []
        for decile in range(1, 11):
            rows.append(
                {
                    "PERMNO": decile,
                    "holding_month": "2001-02",
                    "cnn_decile": decile,
                    "next_month_return": decile / 100.0,
                }
            )
        monthly, _ = build_monthly_portfolios(pd.DataFrame(rows))
        spread = monthly.loc[
            monthly["portfolio"].eq("Q10-Q1"), "return"
        ].iloc[0]
        self.assertTrue(np.isclose(spread, 0.09))


if __name__ == "__main__":
    unittest.main()
