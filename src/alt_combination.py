"""Signal combinations other than rank averaging, on the four-signal (CNN) universe (professor meeting 2026-09-30).

Usage (project root, .venv):  python src/alt_combination.py data/benchmarks/sp500_monthly.csv
Inputs: exactly those of ``longonly_d10_vs_sp500.py`` (stock-level panel from ``build_panel``, CNN scores,
FF5 monthly, S&P 500 CSV). Output: ``outputs/alt_combination/`` (portfolio-level aggregates only).

Pre-specified on 2026-10-02 before running; every design is reported whatever its result.
Methods (definitions in ``alt_combination_core.py``):
  RANK  the current method, rebuilt exactly as ``longonly_d10_vs_sp500`` (CNN only, Equal 25%, CNN 40%, CNN 70%)
  Z     winsorized (1%/99%) z-scores, same three weight sets as RANK (Equal 25%, CNN 40%, CNN 70%)
  RA    (CNN P(up) - month median) / IVOL  (IVOL clipped at 1%/99%)
  FM    Fama-MacBeth expected return on the four Z signals; slopes averaged over the past 60 months (min 24)
  ZLOG  (exploratory, added after the three-signal pilot) Z with log(1+mom), log(beta), log(IVOL); Equal 25%, CNN 70%
Universe: the four-signal matched universe of ``longonly_d10_vs_sp500`` (formation 2004-12..2019-11, holding
2005-01..2019-12, stocks delisting in the formation month excluded). FM is additionally *trained* on earlier
formation months (back to ``TRAIN_FIRST``) where all four signals exist; there the FP beta meets its
750-observation rule but its 60-month window is not yet full. Those months are never evaluated.
Portfolios: decile 10 of the score (EW, VW) and top 50 (EW, VW); VW uses formation-month market cap.
Costs: 0.10% per dollar traded (drift-adjusted, ``top_n_returns``). Metrics as ``longonly_d10_vs_sp500.metrics``
(excess of RF; beta/alpha vs S&P 500 TR) plus 2008 and Mar-May 2009 returns.
Statistics: paired moving-block bootstrap of Sharpe differences (``sharpe_bootstrap``; 6/12-month blocks, 5,000
draws, exploratory, no multiple-comparison adjustment): each Z/ZLOG design vs RANK with the same weights; RA and FM vs
RANK Equal 25% and vs RANK CNN 70%.
Check: the RANK rows must reproduce ``outputs/longonly_d10_vs_sp500/longonly_d10_vs_sp500.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "handoff" / "ivol_combination" / "src"))
from alt_combination_core import (FM_MIN_MONTHS, FM_WINDOW, fama_macbeth_score, fm_summary,  # noqa: E402
                                  risk_adjusted_score, z_score, zlog_score)
from common import assign_deciles  # noqa: E402
from eval_window import FIRST_FORMATION, LAST_FORMATION, START, END, clip  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/alt_combination"
OFFICIAL = ROOT / "outputs/longonly_d10_vs_sp500/longonly_d10_vs_sp500.csv"
COST = 0.001
ALL = 10**7
TRAIN_FIRST = "2001-01"
SIGNALS = [("mom", 1), ("beta", -1), ("ivol_daily", -1), ("cnn_prob_up", 1)]
WEIGHTS = {"CNN only": (0.0, 0.0, 0.0, 1.0), "Equal 25%": (0.25, 0.25, 0.25, 0.25),
           "CNN 40%": (0.2, 0.2, 0.2, 0.4), "CNN 70%": (0.1, 0.1, 0.1, 0.7)}
Z_WEIGHTS = ["Equal 25%", "CNN 40%", "CNN 70%"]
ZLOG_WEIGHTS = ["Equal 25%", "CNN 70%"]
PORTFOLIOS = [("D10", "EW"), ("D10", "VW"), ("Top50", "EW"), ("Top50", "VW")]
OFFICIAL_NAMES = {"RANK CNN only": "CNN only (matched)", "RANK Equal 25%": "Equal 25%", "RANK CNN 40%": "CNN 40%",
                  "RANK CNN 70%": "CNN 70%"}
OFFICIAL_WEIGHTING = {("D10", "EW"): "EW", ("D10", "VW"): "VW", ("Top50", "EW"): "top50 EW"}


def score_designs(four: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Scored evaluation frames per design, plus FM slopes and lambdas."""
    from three_factor_combination import combined_score, pct_rank

    evaluation = four.loc[four["formation_month"].between(FIRST_FORMATION, LAST_FORMATION)].copy()
    g = evaluation.groupby("formation_month")
    ranks = np.column_stack([g["mom"].transform(pct_rank), 1.0 - g["beta"].transform(pct_rank),
                             1.0 - g["ivol_daily"].transform(pct_rank), g["cnn_prob_up"].transform(pct_rank)])
    evaluation["size_pct"] = g["cap"].transform(pct_rank)
    for i, name in enumerate(["mom_pct", "lowbeta_pct", "lowivol_pct", "cnn_pct"]):
        evaluation[name] = ranks[:, i]

    designs = {}
    for name, weights in WEIGHTS.items():  # identical to longonly_d10_vs_sp500
        designs[f"RANK {name}"] = evaluation.assign(score=combined_score(list(ranks.T), weights))
    for name in Z_WEIGHTS:
        designs[f"Z {name}"] = evaluation.assign(score=z_score(evaluation, SIGNALS, WEIGHTS[name]))
    for name in ZLOG_WEIGHTS:
        designs[f"ZLOG {name}"] = evaluation.assign(score=zlog_score(evaluation, SIGNALS, WEIGHTS[name]))
    designs["RA CNN/IVOL"] = evaluation.assign(score=risk_adjusted_score(evaluation, "cnn_prob_up", "ivol_daily"))

    fm, slopes, lambdas = fama_macbeth_score(four, SIGNALS, "ret_final", FM_WINDOW, FM_MIN_MONTHS)
    designs["FM 4 signals"] = evaluation.assign(score=fm.reindex(evaluation.index)).dropna(subset=["score"])
    return designs, slopes, lambdas


def build_portfolios(frame: pd.DataFrame) -> dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]]:
    from top_n_combination import top_n_returns

    d10 = frame.loc[assign_deciles(frame, signal_column="score").astype(int) == 10]
    out = {}
    for size, weighting in PORTFOLIOS:
        source, n = (d10, ALL) if size == "D10" else (frame, 50)
        out[(size, weighting)] = top_n_returns(source, "score", n, weighting)
    return out


def holdings_profile(top: pd.DataFrame) -> dict:
    """Portfolio-weighted average percentile of each characteristic, averaged over months."""
    cols = ["size_pct", "mom_pct", "lowbeta_pct", "lowivol_pct", "cnn_pct"]
    w = top["weight"] / top.groupby("formation_month")["weight"].transform("sum")
    monthly = top[cols].mul(w, axis=0).groupby(top["formation_month"]).sum()
    return {f"avg_{c}": float(monthly[c].mean()) for c in cols}


def compound(r: pd.Series, a: str, b: str) -> float:
    part = r.loc[(r.index >= a) & (r.index <= b)]
    return float(np.prod(1 + part) - 1) if len(part) else np.nan


def run(four: pd.DataFrame, sp: pd.DataFrame, rf: pd.Series, out_dir: Path, official: pd.DataFrame | None) -> dict:
    from longonly_d10_vs_sp500 import metrics
    from sharpe_bootstrap import paired_sharpe_difference

    designs, slopes, lambdas = score_designs(four)
    rows, net, profiles = [], {}, []
    for design, frame in designs.items():
        for (size, weighting), (monthly, top) in build_portfolios(frame).items():
            m = clip(monthly)
            m.index = m.index.astype(str)
            series = {"gross": m["gross"], "net 0.10%": m["gross"] - COST * m["traded"]}
            net[(design, size, weighting)] = series["net 0.10%"]
            for label, r in series.items():
                rows.append({"design": design, "method": design.split()[0], "size": size, "weighting": weighting,
                             "returns": label, "months": len(r), "first_month": r.index.min(),
                             "turnover_one_way": m["traded"].mean() / 2, "stocks": m["holdings"].mean(),
                             "missing_held": int(m["missing_returns"].sum()),
                             **metrics(r, rf.reindex(r.index), sp["sp500tr_ret"].reindex(r.index)),
                             "year_2008": compound(r, "2008-01", "2008-12"),
                             "mar_may_2009": compound(r, "2009-03", "2009-05")})
            profiles.append({"design": design, "size": size, "weighting": weighting, **holdings_profile(top)})
    results = pd.DataFrame(rows)

    comparisons = []
    for size, weighting in PORTFOLIOS:
        pairs = [(f"Z {w}", f"RANK {w}") for w in Z_WEIGHTS] + [(f"ZLOG {w}", f"RANK {w}") for w in ZLOG_WEIGHTS]
        pairs += [(d, b) for d in ["RA CNN/IVOL", "FM 4 signals"] for b in ["RANK Equal 25%", "RANK CNN 70%"]]
        for design, base in pairs:
            a, b = net[(design, size, weighting)], net[(base, size, weighting)]
            comparisons.append({"design": design, "vs": base, "size": size, "weighting": weighting,
                                "common_months": int(len(a.index.intersection(b.index))),
                                **paired_sharpe_difference(*(s.loc[a.index.intersection(b.index)] for s in (a, b)),
                                                           rf)})
    bootstrap = pd.DataFrame(comparisons)

    validation = validate(results, official) if official is not None else pd.DataFrame()
    fm_table = fm_summary(slopes, FIRST_FORMATION, LAST_FORMATION)

    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "alt_combination_results.csv", index=False)
    pd.DataFrame({f"{d} | {s} {w}": r for (d, s, w), r in net.items()}).to_csv(out_dir / "alt_combination_monthly_net.csv")
    bootstrap.to_csv(out_dir / "alt_combination_bootstrap.csv", index=False)
    pd.DataFrame(profiles).to_csv(out_dir / "holdings_profile.csv", index=False)
    slopes.to_csv(out_dir / "fama_macbeth_monthly_slopes.csv")
    lambdas.to_csv(out_dir / "fama_macbeth_lambdas_used.csv")
    fm_table.to_csv(out_dir / "fama_macbeth_summary_2004-12_2019-11.csv", index=False)
    if len(validation):
        validation.to_csv(out_dir / "validation_vs_longonly.csv", index=False)
    return {"results": results, "bootstrap": bootstrap, "validation": validation, "fm": fm_table,
            "lambdas": lambdas, "profiles": pd.DataFrame(profiles)}


def validate(results: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """RANK rows against longonly_d10_vs_sp500.csv (same universe, scores and portfolio code)."""
    rows = []
    for design, name in OFFICIAL_NAMES.items():
        for (size, weighting), label in OFFICIAL_WEIGHTING.items():
            for returns in ["gross", "net 0.10%"]:
                mine = results.query("design == @design and size == @size and weighting == @weighting and returns == @returns")
                theirs = official.query("portfolio == @name and weighting == @label and returns == @returns")
                if len(mine) != 1 or len(theirs) != 1:
                    rows.append({"design": design, "portfolio": f"{size} {weighting}", "returns": returns, "status": "missing"})
                    continue
                diffs = {c: abs(float(mine[c].iloc[0]) - float(theirs[c].iloc[0]))
                         for c in ["annual_mean", "volatility", "sharpe", "max_drawdown", "turnover_one_way"]}
                rows.append({"design": design, "portfolio": f"{size} {weighting}", "returns": returns,
                             "max_abs_diff": max(diffs.values()), "status": "ok" if max(diffs.values()) < 1e-9 else "DIFFERS",
                             **{f"diff_{k}": v for k, v in diffs.items()}})
    return pd.DataFrame(rows)


def main(sp_path: str) -> None:
    from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly
    from three_factor_combination import build_panel
    from top_n_combination import CNN_SCORES

    sp = pd.read_csv(sp_path, dtype={"month": str}).set_index("month")
    rf = load_french_monthly(FF5_MONTHLY_ZIP)["rf"]
    rf.index = rf.index.astype(str)
    scores = pd.read_csv(CNN_SCORES, usecols=["PERMNO", "formation_month", "cnn_prob_up"], dtype={"formation_month": str})
    panel, *_ = build_panel(first=TRAIN_FIRST)
    four = panel.merge(scores, on=["PERMNO", "formation_month"], how="inner", validate="one_to_one")
    official = pd.read_csv(OFFICIAL) if OFFICIAL.exists() else None
    out = run(four, sp, rf, OUTPUT_DIR, official)

    pd.set_option("display.width", 260)
    res = out["results"].query("returns == 'net 0.10%'")
    show = ["design", "size", "weighting", "months", "annual_mean", "volatility", "sharpe", "max_drawdown",
            "turnover_one_way", "alpha_sp", "alpha_sp_t", "year_2008", "mar_may_2009"]
    print(res[show].round(3).to_string(index=False))
    print(out["bootstrap"].round(3).to_string(index=False))
    print(out["fm"].round(4).to_string(index=False))
    if len(out["validation"]):
        bad = out["validation"].query("status != 'ok'")
        print("RANK reproduces longonly_d10_vs_sp500.csv" if bad.empty else f"WARNING: {len(bad)} RANK rows differ:\n{bad}")
    print(f"FM first scored formation month: {out['lambdas'].index.min()} (training starts {four['formation_month'].min()})")


if __name__ == "__main__":
    main(sys.argv[1])
