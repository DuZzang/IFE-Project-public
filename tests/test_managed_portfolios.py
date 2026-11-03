from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from managed_portfolios import Daily, managed_returns  # noqa: E402
from three_factor_combination import combined_score  # noqa: E402


class ManagedPortfolioTests(unittest.TestCase):
    def test_cash_weight_earns_rf_and_turnover_uses_drift(self) -> None:
        holdings = pd.DataFrame({"holding_month": ["2005-01", "2005-01", "2005-02", "2005-02"], "PERMNO": [1, 2, 1, 2],
                                 "w": [0.25] * 4, "ret_final": [0.10, -0.10, 0.0, 0.0]})
        out = managed_returns(holdings, pd.Series({"2005-01": 0.01, "2005-02": 0.01}))
        self.assertAlmostEqual(out.loc["2005-01", "gross"], 0.25 * 0.10 - 0.25 * 0.10 + 0.5 * 0.01)
        self.assertAlmostEqual(out.loc["2005-01", "traded"], 0.5)
        drifted = np.array([0.25 * 1.1, 0.25 * 0.9]) / 1.005
        self.assertAlmostEqual(out.loc["2005-02", "traded"], float(np.abs(0.25 - drifted).sum()))

    def test_leverage_borrows_at_rf(self) -> None:
        # exposure 1.5 (BSC-B without cap): the extra 0.5 is borrowed at RF and traded like any position
        holdings = pd.DataFrame({"holding_month": ["2005-01"], "PERMNO": [1], "w": [1.5], "ret_final": [0.10]})
        out = managed_returns(holdings, pd.Series({"2005-01": 0.01}))
        self.assertAlmostEqual(out.loc["2005-01", "gross"], 1.5 * 0.10 - 0.5 * 0.01)
        self.assertAlmostEqual(out.loc["2005-01", "traded"], 1.5)

    def test_all_cash_month_is_kept_and_reentry_is_traded(self) -> None:
        # stock in January, nothing held in February (all cash), the same stock again in March
        holdings = pd.DataFrame({"holding_month": ["2005-01", "2005-03"], "PERMNO": [1, 1], "w": [1.0, 1.0],
                                 "ret_final": [0.05, 0.02]})
        rf = pd.Series({"2005-01": 0.001, "2005-02": 0.002, "2005-03": 0.003})
        out = managed_returns(holdings, rf, ["2005-01", "2005-02", "2005-03"])
        self.assertEqual(list(out.index), ["2005-01", "2005-02", "2005-03"])
        self.assertAlmostEqual(out.loc["2005-02", "gross"], 0.002)
        self.assertAlmostEqual(out.loc["2005-02", "invested"], 0.0)
        self.assertAlmostEqual(out.loc["2005-02", "traded"], 1.0)  # sell everything
        self.assertAlmostEqual(out.loc["2005-03", "traded"], 1.0)  # buy back in

    def test_daily_path_is_buy_and_hold_and_includes_missing_monthly_returns(self) -> None:
        daily = object.__new__(Daily)
        daily.day_month, daily.rf = np.array([5, 5, 5]), np.zeros(3)
        daily.days = pd.DataFrame({"PERMNO": [1, 1, 1], "day": [0, 1, 2], "mcode": 5, "r": 0.01})
        # stock 2 delisted without daily rows: its monthly return (-50%) is booked on the first or last day
        daily.residuals = pd.DataFrame({"PERMNO": [2], "mcode": [5], "day_end": [2], "day_start": [0], "r": [-0.5]})
        members = pd.DataFrame({"PERMNO": [1, 2], "mcode": [5, 5], "w": [0.5, 0.5]})
        month = 0.5 * 1.01 ** 3 + 0.5 * 0.5 - 1  # formation weights held through the month: -23.48495%
        for residual_day in ["day_end", "day_start"]:
            path = daily.portfolio(members, residual_day=residual_day)
            self.assertAlmostEqual(float((1 + path).prod() - 1), month)
        np.testing.assert_allclose(daily.portfolio(members, rebalance_daily=True).to_numpy(),
                                   [0.01, 0.01, 0.5 * 0.01 + 0.5 * -0.5])  # the round-2 approximation

    def test_trend_holds_only_a_confirmed_uptrend(self) -> None:
        # one month of 250 market days; MA windows end on its last market day (day 249)
        daily = object.__new__(Daily)
        daily.day_month = np.full(250, 24060)
        all_days, gap = np.arange(250), np.r_[np.arange(0, 100), np.arange(145, 250)]  # 45 days missing: 155 of last 200
        rows = [(1, 0.01, all_days), (2, -0.01, all_days), (3, 0.01, np.arange(51, 250)),  # rising, falling, 199 days
                (4, 0.01, gap), (5, 0.0, all_days)]  # rising with a gap (< 80% of 200), flat (MA50 = MA200)
        daily.days = pd.concat([pd.DataFrame({"PERMNO": p, "day": d, "l": step, "mcode": 24060}) for p, step, d in rows],
                               ignore_index=True)
        trend = daily.trend().set_index("PERMNO")["no_uptrend"]
        self.assertEqual(trend.to_dict(), {1: False, 2: True, 3: False, 4: True, 5: True})

    def test_combined_score_does_not_depend_on_array_layout(self) -> None:
        rng = np.random.default_rng(0)
        ranks = rng.random((4, 1000))
        weights = [0.25] * 4
        by_rows = combined_score(list(ranks), weights)
        by_columns = combined_score(list(np.asfortranarray(ranks.T).T), weights)
        np.testing.assert_array_equal(by_rows, by_columns)
        self.assertEqual(combined_score([np.array([0.1]), np.array([0.2])], [0.5, 0.5])[0],
                         combined_score([np.array([0.2]), np.array([0.1])], [0.5, 0.5])[0])


if __name__ == "__main__":
    unittest.main()
