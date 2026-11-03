"""Checks behind the four-signal README numbers (run after ``alt_combination.py``).

Usage (project root, .venv):  python src/alt_combination_checks.py
Reads ``outputs/alt_combination/`` and rebuilds the four-signal panel for the score checks (a few minutes).
Writes month- or portfolio-level aggregates only:
  checks_subperiods.csv     metrics of every net series in 2005-09, 2010-14, 2015-19
  checks_fm_lambdas.csv     average Fama-MacBeth weight per signal (% per month per 1 sd) by formation period
  checks_scores.csv         non-finite scores per design (a NaN score would land in decile 10 of ``assign_deciles``)
  checks_tails.csv          per signal, the monthly max / min / skew of the Z and ZLOG transforms, averaged over months
  checks_concentration.csv  share of top-50 names at a winsorizing bound (Z Equal 25%: momentum at its 99% clip;
                            RA: IVOL at its 1% floor)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "handoff" / "ivol_combination" / "src"))
import alt_combination as ac  # noqa: E402
import alt_combination_core as core  # noqa: E402

OUT = ac.OUTPUT_DIR
PERIODS = [("2005-01", "2009-12"), ("2010-01", "2014-12"), ("2015-01", "2019-12")]
LAMBDA_PERIODS = [("2004-12", "2007-12"), ("2008-01", "2010-12"), ("2011-01", "2014-12"), ("2015-01", "2019-11")]
NAMES = [c for c, _ in ac.SIGNALS]


def top50(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values(["formation_month", "score", "PERMNO"], ascending=[True, False, True])
    return ranked.groupby("formation_month").head(50)


def at_bound(frame: pd.DataFrame, column: str, q: float) -> float:
    bound = frame.groupby("formation_month")[column].transform(lambda v: v.quantile(q))
    top = top50(frame)
    hit = top[column] >= bound.loc[top.index] if q > 0.5 else top[column] <= bound.loc[top.index]
    return float(hit.mean())


def main() -> None:
    from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly
    from three_factor_combination import build_panel
    from top_n_combination import CNN_SCORES

    rf = load_french_monthly(FF5_MONTHLY_ZIP)["rf"]
    rf.index = rf.index.astype(str)
    net = pd.read_csv(OUT / "alt_combination_monthly_net.csv", index_col=0)
    net.index = net.index.astype(str)
    subperiods = pd.DataFrame([{"series": c, "period": f"{a}..{b}", **core.basic_metrics(net[c].loc[a:b], rf)}
                               for c in net for a, b in PERIODS])
    lam = pd.read_csv(OUT / "fama_macbeth_lambdas_used.csv", index_col=0)
    lam.index = lam.index.astype(str)
    lambdas = pd.DataFrame([{"formation": f"{a}..{b}", **(100 * lam.loc[a:b, NAMES].mean()).to_dict(),
                             "months_used_min": int(lam.loc[a:b, "months_used"].min())} for a, b in LAMBDA_PERIODS])

    scores = pd.read_csv(CNN_SCORES, usecols=["PERMNO", "formation_month", "cnn_prob_up"], dtype={"formation_month": str})
    panel, *_ = build_panel(first=ac.TRAIN_FIRST)
    four = panel.merge(scores, on=["PERMNO", "formation_month"], how="inner", validate="one_to_one")
    designs, *_ = ac.score_designs(four)
    checks = pd.DataFrame([{"design": d, "rows": len(f), "nonfinite": int((~np.isfinite(f["score"].to_numpy(float))).sum())}
                           for d, f in designs.items()])
    ev = designs["RANK Equal 25%"]
    tails = []
    for how in ["z", "zlog"]:
        t = core.transformed(ev, ac.SIGNALS, how).groupby(ev["formation_month"])
        for stat, frame in [("max", t.max()), ("min", t.min()), ("skew", t.skew())]:
            tails.append({"transform": how, "monthly": stat, **frame.mean().to_dict()})
    concentration = pd.DataFrame([
        {"design": "Z Equal 25%", "bound": "momentum at monthly 99% clip", "top50_share": at_bound(designs["Z Equal 25%"], "mom", 0.99)},
        {"design": "RA CNN/IVOL", "bound": "IVOL at monthly 1% floor", "top50_share": at_bound(designs["RA CNN/IVOL"], "ivol_daily", 0.01)},
    ])

    for name, frame in [("subperiods", subperiods), ("fm_lambdas", lambdas), ("scores", checks),
                        ("tails", pd.DataFrame(tails)), ("concentration", concentration)]:
        frame.to_csv(OUT / f"checks_{name}.csv", index=False)
        print(f"== {name}\n{frame.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
