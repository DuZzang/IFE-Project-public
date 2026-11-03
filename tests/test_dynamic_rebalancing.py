from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynamic_rebalancing import select_path  # noqa: E402


class DynamicRebalancingTests(unittest.TestCase):
    def test_band_keeps_last_months_names_and_team_rule_drops_downtrends(self) -> None:
        # 20 stocks -> decile 10 holds 2. Month 1: stocks 1 and 2 are best. Month 2: stock 3 leads, stock 1 falls to
        # rank 4 (inside the 2N = 4 band) and is in a downtrend.
        first = pd.DataFrame({"formation_month": "2005-01", "PERMNO": range(1, 21), "score": [20 - i for i in range(20)],
                              "no_uptrend": False})
        second = pd.DataFrame({"formation_month": "2005-02", "PERMNO": range(1, 21),
                               "score": [97, 99, 100, 98] + [0] * 16, "no_uptrend": [True] + [False] * 19})
        frame = pd.concat([first, second], ignore_index=True)
        held = {kind: set(select_path(frame, "D10", kind, 2.0)["2005-02"][0]) for kind in ["base", "band", "team", "cash"]}
        self.assertEqual(held, {"base": {2, 3}, "band": {1, 2}, "team": {2, 3}, "cash": {2, 3}})


if __name__ == "__main__":
    unittest.main()
