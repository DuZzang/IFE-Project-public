"""Evaluate every strategy on the common holding window 2005-01 to 2019-12 (``eval_window``).

The start is set by the FP beta, which needs five years of history (first formation 2004-12); the end
by the CNN images, which stop at the 2019-12 formation month.
Signals and deciles are unchanged; only the evaluation months are restricted. The combination,
top-50, FP BAB and market results use the same window and are read from outputs/team_report.
Only portfolio-level aggregates are written.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import plot_delisting_deciles as deciles_plot  # noqa: E402
import plot_ivol_factor_alphas as alphas_plot  # noqa: E402
from delisting_robustness import CNN, IVOL, SUPPLEMENT, monthly_portfolios  # noqa: E402
from eval_window import END, FIRST_FORMATION, LAST_FORMATION, PERIOD_TEXT, START  # noqa: E402
from ivol_specifications import (  # noqa: E402
    ASSIGNMENTS,
    DESIGNS,
    FF3_MONTHLY_ZIP,
    FF5_MONTHLY_ZIP,
    assign_groups,
    formation_prices,
    load_french_monthly,
    spread_series,
)
from momentum_lowbeta_deciles import (  # noqa: E402
    CRSP_MONTHLY,
    attach_outcomes,
    beta_signal,
    load_prices,
    momentum_signals,
    monthly_market,
)
from professor_universe import load_formation_caps  # noqa: E402
from team_report_metrics import name_turnover, perf  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/common_window_2005"
PERIODS = {
    "IVOL": PERIOD_TEXT[0].upper() + PERIOD_TEXT[1:],
    "MOM": "Signal P(t-1)/P(t-12) - 1 on exact calendar months. " + PERIOD_TEXT[0].upper() + PERIOD_TEXT[1:],
    "LOWBETA": "60-month CAPM beta vs Fama-French market (at least 48 months); a decile sort, not the beta-neutral FP BAB. "
               + PERIOD_TEXT[0].upper() + PERIOD_TEXT[1:],
    "CNN": PERIOD_TEXT[0].upper() + PERIOD_TEXT[1:],
}
LABEL_XY = {"IVOL": (5.3, 0.02)}  # keep the before/after label inside the axes


def clip(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    index = frame.index.astype(str)
    return frame.loc[(index >= START) & (index <= END)]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ff3, ff5 = load_french_monthly(FF3_MONTHLY_ZIP), load_french_monthly(FF5_MONTHLY_ZIP)
    supplement = pd.read_csv(
        SUPPLEMENT, usecols=["PERMNO", "formation_month", "ret_adjprc", "ret_final", "ret_source"],
        dtype={"formation_month": str},
    )
    metrics, decile_rows, spread_rows, delist_rows, turnover_rows = [], [], [], [], []

    def evaluate(label: str, frame: pd.DataFrame, long: int, short: int) -> None:
        for version, column in [("adj_prc_only", "ret_adjprc_eval"), ("with_delisting", "ret_final")]:
            for weighting, weighted in [("EW", False), ("VW", True)]:
                wide = clip(monthly_portfolios(frame, column, weighted))
                spread = wide[long] - wide[short]
                p = perf(spread, ff3, ff5, long_only=False)
                metrics.append({"strategy": label, "version": version, "weighting": weighting,
                                "portfolio": f"Q{long}-Q{short}", **p})
                spread_rows.append({"strategy": label, "weighting": weighting, "version": version,
                                    "spread": f"Q{long}-Q{short}", "annualized_mean": p["annual_mean"],
                                    "newey_west_t_lag4": p["nw_t"]})
                for decile in range(1, 11):
                    decile_rows.append({"strategy": label, "weighting": weighting, "version": version, "decile": decile,
                                        "annualized_mean": 12 * wide[decile].mean(),
                                        "annualized_volatility": math.sqrt(12) * wide[decile].std(ddof=1)})
        window = frame.loc[frame["holding_month"].between(START, END)]
        for decile in [1, 10]:
            part = window.loc[(window["decile"] == decile) & (window["ret_source"] == "crsp_delisting")]
            delist_rows.append({"strategy": label, "decile": decile, "delisting_replacements": int(len(part)),
                                "mean_delisting_return": float(part["ret_final"].mean())})
        formed = frame.loc[frame["formation_month"].between(FIRST_FORMATION, LAST_FORMATION)]
        turnover_rows.append({"strategy": label, "long_decile": long,
                              "long_leg_name_turnover": name_turnover(formed, long),
                              "short_leg_name_turnover": name_turnover(formed, short)})

    # IVOL and CNN
    ivol = pd.read_csv(
        IVOL, usecols=["PERMNO", "formation_month", "decile", "next_month", "next_month_return", "formation_market_cap"],
        dtype={"formation_month": str, "next_month": str},
    ).rename(columns={"next_month": "holding_month", "next_month_return": "ret_adjprc_eval", "formation_market_cap": "cap"})
    ivol = ivol.merge(supplement[["PERMNO", "formation_month", "ret_final", "ret_source"]], on=["PERMNO", "formation_month"], how="left")
    evaluate("IVOL", ivol, 1, 10)
    cnn = pd.read_csv(
        CNN, usecols=["PERMNO", "formation_month", "holding_month", "cnn_decile", "next_month_return", "formation_market_cap"],
        dtype={"formation_month": str, "holding_month": str},
    ).rename(columns={"cnn_decile": "decile", "next_month_return": "ret_adjprc_eval", "formation_market_cap": "cap"})
    cnn = cnn.merge(supplement[["PERMNO", "formation_month", "ret_final", "ret_source"]], on=["PERMNO", "formation_month"], how="left")
    evaluate("CNN", cnn, 10, 1)

    # Momentum (standard and notebook definition) and monthly-beta deciles
    prices, universe = load_prices()
    signals = momentum_signals(prices, universe)
    signals["LOWBETA"] = beta_signal(prices, universe, monthly_market())
    keys = pd.concat([signals[k][["PERMNO", "formation_month"]] for k in signals]).drop_duplicates()
    caps, _ = load_formation_caps(CRSP_MONTHLY, keys, chunksize=500_000)
    evaluate("MOM", attach_outcomes(signals["MOM"], supplement, caps), 10, 1)
    evaluate("MOM_t2_variant", attach_outcomes(signals["MOM_t2_variant"], supplement, caps), 10, 1)
    evaluate("LOWBETA", attach_outcomes(signals["LOWBETA"], supplement, caps), 1, 10)

    # Results that already start in 2005-01
    existing = pd.read_csv(ROOT / "outputs/team_report/strategy_metrics.csv")
    keep = existing.loc[existing["strategy"].isin(["FP_BAB", "COMBINED_3", "P  top50 EW all".strip(),
                                                   "C4 top50 EW 4 signals", "Market (FF Mkt)", "Universe EW"])]
    metrics_frame = pd.concat([pd.DataFrame(metrics), keep], ignore_index=True)

    # IVOL eight designs on the common window
    frame = pd.read_csv(ASSIGNMENTS, usecols=["PERMNO", "formation_month", "EXCHCD", "ivol_daily", "formation_market_cap"],
                        dtype={"formation_month": str})
    frame = frame.merge(supplement[["PERMNO", "formation_month", "ret_final"]], on=["PERMNO", "formation_month"], how="left")
    frame = frame.assign(holding_month=(pd.PeriodIndex(frame["formation_month"], freq="M") + 1).astype(str))
    frame = formation_prices(frame)
    design_rows = []
    for design_id, label, weighting, groups, stock_filter, breakpoints in DESIGNS:
        sample = frame.loc[frame["price"].ge(5.0)] if stock_filter == "price5" else frame
        sample = sample.assign(group=assign_groups(sample, groups, breakpoints))
        spread, _ = spread_series(sample, weighting, groups)
        spread = clip(spread)
        p = perf(spread, ff3, ff5, long_only=False)
        x = np.column_stack([np.ones(len(spread)), ff5.reindex(spread.index)[["mkt_rf", "smb", "hml", "rmw", "cma"]].to_numpy()])
        betas = np.linalg.lstsq(x, spread.to_numpy(), rcond=None)[0][1:]
        design_rows.append({"design": design_id, "label": label, **p,
                            **dict(zip(["ff5_beta_mkt", "ff5_beta_smb", "ff5_beta_hml", "ff5_beta_rmw", "ff5_beta_cma"], betas))})

    # IVOL decile alphas and the volatility ratio on the common window
    alpha_rows, factors = [], {"FF3": ff3, "FF5": ff5}
    columns = {"FF3": ["mkt_rf", "smb", "hml"], "FF5": ["mkt_rf", "smb", "hml", "rmw", "cma"]}
    for sample_key in ["all", "price5"]:
        sample = frame.loc[frame["price"].ge(5.0)] if sample_key == "price5" else frame
        sample = sample.assign(group=assign_groups(sample, 10, "all"))
        for weighting in ["EW", "VW"]:
            wide = clip(alphas_plot.decile_returns(sample, weighting))
            for model in ["FF3", "FF5"]:
                rf = factors[model]["rf"].reindex(wide.index)
                for decile in range(1, 11):
                    a, t = alphas_plot.fit(wide[decile] - rf, factors[model], columns[model])
                    alpha_rows.append((sample_key, weighting, model, str(decile), a, t))
                a, t = alphas_plot.fit(wide[1] - wide[10], factors[model], columns[model])
                alpha_rows.append((sample_key, weighting, model, "Q1-Q10", a, t))
            if sample_key == "all":
                q1, q10 = wide[1].to_numpy(), wide[10].to_numpy()
                rng, n, block, ratios = np.random.default_rng(0), len(q1), 12, []
                for _ in range(2000):
                    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
                    sel = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
                    ratios.append(q10[sel].std(ddof=1) / q1[sel].std(ddof=1))
                lo, hi = np.percentile(ratios, [2.5, 97.5])
                metrics_frame = pd.concat([metrics_frame, pd.DataFrame([{
                    "strategy": "IVOL_vol_ratio", "version": "with_delisting", "weighting": weighting,
                    "portfolio": "Q10/Q1 SD", "annual_mean": q10.std(ddof=1) / q1.std(ddof=1), "volatility": lo, "sharpe": hi}])],
                    ignore_index=True)
    alphas = pd.DataFrame(alpha_rows, columns=["sample", "weighting", "model", "portfolio", "alpha_annual", "alpha_t"])

    metrics_frame.to_csv(OUTPUT_DIR / "strategy_metrics_2005.csv", index=False)
    pd.DataFrame(decile_rows).to_csv(OUTPUT_DIR / "decile_summary_2005.csv", index=False)
    pd.DataFrame(spread_rows).to_csv(OUTPUT_DIR / "spread_summary_2005.csv", index=False)
    pd.DataFrame(delist_rows).to_csv(OUTPUT_DIR / "delisting_by_decile_2005.csv", index=False)
    pd.DataFrame(turnover_rows).to_csv(OUTPUT_DIR / "turnover_2005.csv", index=False)
    pd.DataFrame(design_rows).to_csv(OUTPUT_DIR / "ivol_designs_2005.csv", index=False)
    alphas.to_csv(OUTPUT_DIR / "ivol_decile_alphas_2005.csv", index=False)

    deciles = pd.DataFrame(decile_rows)
    spreads = pd.DataFrame(spread_rows)
    for label in ["IVOL", "CNN", "MOM", "LOWBETA"]:
        deciles_plot.draw(label, deciles, spreads, spec_override={"period": PERIODS[label], "file": f"{label.lower()}_deciles_2005.png",
                                                        **({"label_xy": LABEL_XY[label]} if label in LABEL_XY else {})},
                          output_dir=OUTPUT_DIR)
    for sample_key in ["all", "price5"]:
        alphas_plot.draw(alphas, sample_key, period=PERIOD_TEXT,
                         output_dir=OUTPUT_DIR)

    pd.set_option("display.width", 260)
    show = ["strategy", "version", "weighting", "portfolio", "start", "end", "months", "annual_mean", "volatility", "sharpe",
            "nw_t", "max_drawdown", "capm_alpha", "capm_t", "ff3_alpha", "ff3_t", "ff5_alpha", "ff5_t"]
    print(metrics_frame[show].round(3).to_string(index=False))
    print(pd.DataFrame(design_rows)[["design", "label", "annual_mean", "nw_t", "capm_alpha", "capm_t", "ff3_alpha", "ff3_t",
                                     "ff5_alpha", "ff5_t", "ff5_beta_rmw"]].round(3).to_string(index=False))
    print(pd.DataFrame(delist_rows).round(3).to_string(index=False))
    print(pd.DataFrame(turnover_rows).round(3).to_string(index=False))
    dec = deciles.pivot_table(index=["strategy", "weighting", "version"], columns="decile", values="annualized_mean") * 100
    print(dec.round(1).to_string())
    cell = alphas.assign(v=lambda d: (100 * d["alpha_annual"]).round(1).astype(str) + " (" + d["alpha_t"].round(2).astype(str) + ")")
    print(cell.pivot_table(index=["sample", "weighting", "model"], columns="portfolio", values="v", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
