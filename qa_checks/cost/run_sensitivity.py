"""Q&A sensitivity of the original four-signal Top 50 EW portfolio.

Read existing project data and modules; write only portfolio-level outputs here.
CNN weights and cost rates are exploratory, not an out-of-sample selection test.
"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2],
                    help="Extracted project root containing src/, derived/, handoff/ and outputs/.")
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results",
                    help="New directory for results; existing result files are never overwritten.")
args = parser.parse_args()
PROJECT = args.project_root.expanduser().resolve()
OUT = args.output_dir.expanduser().resolve()
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(1, str(PROJECT / "handoff/ivol_combination/src"))

from three_factor_combination import build_panel, pct_rank, combined_score
from top_n_combination import CNN_SCORES, top_n_returns
from eval_window import clip
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly
from longonly_d10_vs_sp500 import metrics


def main() -> None:
    expected_names = ["top50_cnn70_cost_sensitivity.csv", "top50_cnn_weight_sensitivity.csv",
                      "top50_cnn70_cost_sensitivity_monthly.csv", "verification.json"]
    if any((OUT / name).exists() for name in expected_names):
        raise FileExistsError(f"Results already exist in {OUT}; choose a new --output-dir")
    OUT.mkdir(parents=True, exist_ok=True)
    saved_path = PROJECT / "outputs/longonly_d10_vs_sp500/longonly_d10_monthly_net.csv"
    sp_path = PROJECT / "data/benchmarks/sp500_monthly.csv"
    saved = pd.read_csv(saved_path, dtype={"holding_month": str}).set_index("holding_month")
    rf = load_french_monthly(FF5_MONTHLY_ZIP)["rf"]
    rf.index = rf.index.astype(str)
    sp = pd.read_csv(sp_path, dtype={"month": str}).set_index("month")["sp500tr_ret"]
    print("Building matched stock-month panel from existing project inputs...", flush=True)
    panel, *_ = build_panel()
    scores = pd.read_csv(CNN_SCORES, usecols=["PERMNO", "formation_month", "cnn_prob_up"], dtype={"formation_month": str})
    four = panel.merge(scores, on=["PERMNO", "formation_month"], how="inner", validate="one_to_one")
    del panel, scores
    groups = four.groupby("formation_month")
    ranks = [groups["mom"].transform(pct_rank), 1.0 - groups["beta"].transform(pct_rank),
             1.0 - groups["ivol_daily"].transform(pct_rank), groups["cnn_prob_up"].transform(pct_rank)]
    percentages = [25, 40, 50, 60, 70, 80, 100]
    monthly_by_weight = {}
    weight_rows = []
    checks = {}
    for pct in percentages:
        weight = pct / 100
        remaining = (1 - weight) / 3
        scored = four.assign(score=combined_score(ranks, [remaining, remaining, remaining, weight]))
        monthly = clip(top_n_returns(scored, "score", 50, "EW")[0])
        monthly.index = monthly.index.astype(str)
        if len(monthly) != 180 or monthly.index[0] != "2005-01" or monthly.index[-1] != "2019-12":
            raise ValueError(f"Unexpected evaluation window for {pct}%: {len(monthly)}")
        if monthly.index.duplicated().any() or monthly[["gross", "traded"]].isna().any().any():
            raise ValueError("Duplicated or missing monthly gross/traded values")
        monthly_by_weight[pct] = monthly
        net = monthly["gross"] - .001 * monthly["traded"]
        original = {25: "Equal 25% | top50 EW", 40: "CNN 40% | top50 EW", 70: "CNN 70% | top50 EW"}.get(pct)
        if original:
            expected = saved[original].reindex(monthly.index)
            diff = float((net - expected).abs().max())
            checks[original] = {"max_abs_monthly_difference": diff, "months": len(monthly)}
            if expected.isna().any() or diff > 1e-10:
                raise ValueError(f"Original monthly output does not match: {original}: {diff}")
        m = metrics(net, rf.reindex(monthly.index), sp.reindex(monthly.index))
        weight_rows.append({"cnn_weight": weight, "each_other_weight": remaining, "months": len(monthly),
                            "one_way_turnover": monthly["traded"].mean() / 2,
                            "missing_held": int(monthly["missing_returns"].sum()), **m})
        print(f"CNN weight {pct}% completed", flush=True)

    objective = monthly_by_weight[70]
    cost_rows, monthly_rows = [], []
    for cost in [0.0, .0005, .001, .0015, .002, .003, .005]:
        net = objective["gross"] - cost * objective["traded"]
        m = metrics(net, rf.reindex(objective.index), sp.reindex(objective.index))
        cost_rows.append({"cost_per_traded_dollar": cost, "cost_basis_points": cost*10000,
                          "months": len(net), "one_way_turnover": objective["traded"].mean()/2, **m})
        monthly_rows.append(pd.DataFrame({"holding_month": objective.index, "cost_per_traded_dollar": cost,
                                          "gross_return": objective["gross"].to_numpy(), "two_way_traded": objective["traded"].to_numpy(),
                                          "net_return": net.to_numpy()}))

    pd.DataFrame(cost_rows).to_csv(OUT / "top50_cnn70_cost_sensitivity.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUT / "top50_cnn_weight_sensitivity.csv", index=False)
    pd.concat(monthly_rows, ignore_index=True).to_csv(OUT / "top50_cnn70_cost_sensitivity_monthly.csv", index=False)
    (OUT / "verification.json").write_text(json.dumps({
        "period": "2005-01 to 2019-12 holding months", "months": 180,
        "monthly_reproduction_checks": checks,
        "cost_formula": "gross - cost_per_traded_dollar * sum(abs(target_weight - prior_drifted_weight))",
        "one_way_turnover": "two-way traded / 2", "evaluation": "arithmetic annual return=12*mean; volatility=sqrt(12)*sample std; Sharpe uses monthly RF excess returns; max drawdown includes initial wealth",
        "data": ["derived/professor_universe_2000_2020", "handoff/금공개 팀플/IFE_term_stock_data.csv", "handoff/금공개 팀플/IVOL/CRSP_Monthly_1999-12_2020.csv.gz", "handoff/delisting_supplement/next_month_returns_with_delisting_2000-2020.csv.gz", "outputs/cnn_oos_deciles_2001_2019/cnn_signal_scores_2001_2019.csv.gz", "data/benchmarks/sp500_monthly.csv", "F-F_Research_Data_5_Factors_2x3_CSV.zip"],
        "limitations": ["weights selected and checked on same 2005-2019 sample", "costs fixed per dollar and no market impact", "missing held returns treated by existing top_n_returns policy", "new 50/60/80 weights are exploratory"]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(cost_rows)[["cost_basis_points", "annual_mean", "volatility", "sharpe", "max_drawdown"]].to_string(index=False), flush=True)
    print(pd.DataFrame(weight_rows)[["cnn_weight", "annual_mean", "volatility", "sharpe", "max_drawdown"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
