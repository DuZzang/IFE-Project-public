"""Tests for alt_combination_core and the alt_combination runner (synthetic data only; no CRSP or CNN files)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "handoff" / "ivol_combination" / "src"))

import alt_combination_core as core  # noqa: E402

MONTHS = [str(p) for p in pd.period_range("2002-11", "2019-11", freq="M")]


def synthetic_panel(n_stocks: int = 120, months=MONTHS, seed: int = 7) -> pd.DataFrame:
    """Four signals; next-month return loads on CNN (+) and beta (-) so FM has something to find."""
    rng = np.random.default_rng(seed)
    frames = []
    for m in months:
        mom = rng.normal(0.1, 0.4, n_stocks)
        beta = rng.lognormal(0.0, 0.3, n_stocks)
        ivol = rng.lognormal(-3.9, 0.5, n_stocks)
        cnn = rng.normal(0.5, 0.03, n_stocks)
        ret = 0.004 * (cnn - 0.5) / 0.03 - 0.002 * (beta - 1.0) / 0.3 + rng.normal(0, 0.06, n_stocks)
        ret[rng.random(n_stocks) < 0.01] = np.nan  # a few missing next-month returns
        frames.append(pd.DataFrame({"PERMNO": np.arange(10001, 10001 + n_stocks), "formation_month": m,
                                    "holding_month": str(pd.Period(m, freq="M") + 1), "mom": mom, "beta": beta,
                                    "ivol_daily": ivol, "cnn_prob_up": cnn, "cap": rng.lognormal(6, 1.5, n_stocks),
                                    "ret_final": ret}))
    return pd.concat(frames, ignore_index=True)


SIGNALS = [("mom", 1), ("beta", -1), ("ivol_daily", -1), ("cnn_prob_up", 1)]


class EquivalenceWithTeamCode(unittest.TestCase):
    """The core copies must behave exactly like the team functions they mirror."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = synthetic_panel(months=MONTHS[:30])

    def test_rank_score_equals_combined_score(self) -> None:
        from three_factor_combination import combined_score, pct_rank

        g = self.panel.groupby("formation_month")
        ranks = np.column_stack([g["mom"].transform(pct_rank), 1.0 - g["beta"].transform(pct_rank),
                                 1.0 - g["ivol_daily"].transform(pct_rank), g["cnn_prob_up"].transform(pct_rank)])
        for weights in [(0.25,) * 4, (0.1, 0.1, 0.1, 0.7), (0.2, 0.2, 0.2, 0.4)]:
            np.testing.assert_array_equal(core.rank_score(self.panel, SIGNALS, weights), combined_score(list(ranks.T), weights))

    def test_deciles_and_portfolios_equal_team(self) -> None:
        from common import assign_deciles
        from top_n_combination import top_n_returns

        frame = self.panel.assign(score=core.rank_score(self.panel, SIGNALS, (0.25,) * 4))
        np.testing.assert_array_equal(core.assign_deciles(frame, "score").to_numpy(),
                                      assign_deciles(frame, signal_column="score").to_numpy())
        for n, weighting in [(50, "EW"), (50, "VW"), (12, "EW")]:
            mine, _ = core.top_n_returns(frame, "score", n, weighting)
            theirs, _ = top_n_returns(frame, "score", n, weighting)
            pd.testing.assert_frame_equal(mine, theirs)

    def test_bootstrap_equals_team(self) -> None:
        from sharpe_bootstrap import paired_sharpe_difference

        rng = np.random.default_rng(1)
        idx = [str(p) for p in pd.period_range("2005-01", periods=180, freq="M")]
        a, b = (pd.Series(rng.normal(0.01, 0.04, 180), index=idx) for _ in range(2))
        rf = pd.Series(0.001, index=idx)
        mine, theirs = core.paired_sharpe_difference(a, b, rf), paired_sharpe_difference(a, b, rf)
        for k, v in theirs.items():
            self.assertAlmostEqual(mine[k], v, places=12)


class Methods(unittest.TestCase):
    def test_zscore_winsorizes_standardizes_and_flips(self) -> None:
        values = list(range(1, 1000)) + [100_000]  # one huge outlier in a 1,000-stock month
        frame = pd.DataFrame({"formation_month": "2010-01", "PERMNO": range(1000), "x": values, "y": values})
        z = core.transformed(frame, [("x", 1), ("y", -1)], "z")
        self.assertAlmostEqual(z["x"].mean(), 0.0, places=12)
        self.assertAlmostEqual(z["x"].std(ddof=1), 1.0, places=12)
        self.assertLess(z["x"].max(), 2.0)  # outlier clipped to the 99th percentile, not ~10 sd away
        self.assertAlmostEqual(z["x"].iloc[-1], z["x"].iloc[-10], places=12)  # top 1% share the clipped value
        np.testing.assert_allclose(z["y"], -z["x"])

    def test_zlog_is_z_of_logged_winsorized_values(self) -> None:
        rng = np.random.default_rng(2)
        frame = pd.DataFrame({"formation_month": "2010-01", "PERMNO": range(500), "mom": rng.lognormal(0, 0.8, 500) - 1,
                              "ivol_daily": rng.lognormal(-4, 0.6, 500), "cnn_prob_up": rng.normal(0.5, 0.03, 500)})
        z = core.transformed(frame, [("mom", 1), ("ivol_daily", -1), ("cnn_prob_up", 1)], "zlog")
        x = np.log(core.winsorize(frame["ivol_daily"]))
        np.testing.assert_allclose(z["ivol_daily"], -(x - x.mean()) / x.std(ddof=1))
        m = np.log1p(core.winsorize(frame["mom"]))
        np.testing.assert_allclose(z["mom"], (m - m.mean()) / m.std(ddof=1))
        np.testing.assert_allclose(z["cnn_prob_up"], core.transformed(frame, [("cnn_prob_up", 1)], "z")["cnn_prob_up"])
        self.assertLess(abs(z["ivol_daily"].skew()), abs(core.transformed(frame, [("ivol_daily", -1)], "z")["ivol_daily"].skew()))

    def test_risk_adjusted_by_hand(self) -> None:
        frame = pd.DataFrame({"formation_month": "2010-01", "PERMNO": range(5), "p": [0.50, 0.52, 0.54, 0.56, 0.58],
                              "ivol": [0.01, 0.02, 0.02, 0.04, 0.02]})
        lo, hi = frame["ivol"].quantile([0.01, 0.99])
        expected = (frame["p"] - 0.54) / frame["ivol"].clip(lo, hi)
        np.testing.assert_allclose(core.risk_adjusted_score(frame, "p", "ivol"), np.round(expected, 12))
        self.assertEqual(int(np.argmax(core.risk_adjusted_score(frame, "p", "ivol"))), 4)

    def test_fama_macbeth_recovers_premia_and_has_no_lookahead(self) -> None:
        panel = synthetic_panel(n_stocks=400, months=MONTHS[:60], seed=3)
        score, slopes, lam = core.fama_macbeth_score(panel, SIGNALS, "ret_final")
        # first score needs 24 past months: formation month index 24 (2004-11)
        self.assertEqual(lam.index.min(), MONTHS[24])
        self.assertTrue(score[panel["formation_month"] < MONTHS[24]].isna().all())
        avg = slopes[["mom", "beta", "ivol_daily", "cnn_prob_up"]].mean()
        self.assertAlmostEqual(avg["cnn_prob_up"], 0.004, delta=0.0008)
        self.assertAlmostEqual(avg["beta"], 0.002, delta=0.0008)  # beta enters as -z, so the premium is positive
        self.assertLess(abs(avg["mom"]), 0.0008)

        t = MONTHS[40]
        at_t = panel["formation_month"] == t
        changed_now = panel.copy()
        changed_now.loc[at_t, "ret_final"] = 5.0  # return realized after t: must not move the month-t score
        s_now, *_ = core.fama_macbeth_score(changed_now, SIGNALS, "ret_final")
        np.testing.assert_array_equal(s_now[at_t].to_numpy(), score[at_t].to_numpy())
        changed_past = panel.copy()
        changed_past.loc[panel["formation_month"] == MONTHS[39], "ret_final"] *= -3  # known at t: should matter
        s_past, *_ = core.fama_macbeth_score(changed_past, SIGNALS, "ret_final")
        self.assertFalse(np.allclose(s_past[at_t].to_numpy(), score[at_t].to_numpy()))

    def test_lambdas_use_at_most_60_months(self) -> None:
        panel = synthetic_panel(n_stocks=60, months=MONTHS[:90], seed=5)
        _, _, lam = core.fama_macbeth_score(panel, SIGNALS, "ret_final")
        self.assertEqual(int(lam["months_used"].max()), 60)
        self.assertEqual(int(lam["months_used"].min()), 24)


class RunnerDryRun(unittest.TestCase):
    def test_runner_end_to_end_on_synthetic_inputs(self) -> None:
        import alt_combination as runner

        four = synthetic_panel()
        months = [str(p) for p in pd.period_range("2005-01", "2019-12", freq="M")]
        rng = np.random.default_rng(11)
        sp = pd.DataFrame({"sp500tr_ret": rng.normal(0.008, 0.04, len(months))}, index=months)
        rf = pd.Series(0.001, index=months)
        with tempfile.TemporaryDirectory() as tmp:
            out = runner.run(four, sp, rf, Path(tmp), official=None)
            files = {p.name for p in Path(tmp).iterdir()}
        res = out["results"]
        designs = set(res["design"])
        self.assertEqual(len(designs), 4 + 3 + 2 + 1 + 1)
        self.assertEqual(len(res), len(designs) * 4 * 2)
        self.assertTrue((res["months"] == 180).all(), res.groupby("design")["months"].min())
        self.assertEqual(out["lambdas"].index.min(), "2004-11")
        self.assertEqual(len(out["bootstrap"]), 4 * (3 + 2 + 4))
        self.assertIn("alt_combination_results.csv", files)
        self.assertTrue((out["profiles"]["avg_size_pct"].between(0, 1)).all())

        # validate(): a copy of the RANK rows renamed like longonly_d10_vs_sp500.csv must pass, a changed one must not
        official = res[res["design"].str.startswith("RANK") & ~((res["size"] == "Top50") & (res["weighting"] == "VW"))].copy()
        official["portfolio"] = official["design"].map(runner.OFFICIAL_NAMES)
        official["weighting"] = [runner.OFFICIAL_WEIGHTING[(s, w)] for s, w in zip(official["size"], official["weighting"])]
        check = runner.validate(res, official)
        self.assertTrue((check["status"] == "ok").all())
        official.loc[official.index[0], "sharpe"] += 0.01
        self.assertEqual(int((runner.validate(res, official)["status"] == "DIFFERS").sum()), 1)


if __name__ == "__main__":
    unittest.main()
