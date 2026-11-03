"""How much of the CNN decile spread is short-term reversal?

Regresses the delisting-corrected CNN Q10-Q1 spread (EW and VW, holding months
2001-02 to 2020-01) on CAPM / FF3 / FF5, each with and without the Fama-French
short-term reversal factor (ST_Rev, formed on month-t returns and held in t+1, the
same timing as the CNN spread). Only portfolio-level aggregates are written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from delisting_robustness import CNN, SUPPLEMENT, monthly_portfolios  # noqa: E402
from ivol_specifications import FACTOR_DIR, load_french_monthly  # noqa: E402
from performance_metrics import newey_west_t_stat, ols_alpha_hac  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/cnn_reversal_check"
MODELS = {
    "CAPM": ["mkt_rf"],
    "CAPM + ST_Rev": ["mkt_rf", "st_rev"],
    "FF3": ["mkt_rf", "smb", "hml"],
    "FF3 + ST_Rev": ["mkt_rf", "smb", "hml", "st_rev"],
    "FF5": ["mkt_rf", "smb", "hml", "rmw", "cma"],
    "FF5 + ST_Rev": ["mkt_rf", "smb", "hml", "rmw", "cma", "st_rev"],
}


def main(start: str | None = None, end: str | None = None) -> None:
    """``start``/``end`` (YYYY-MM) restrict holding months, e.g. 2005-01 2019-12 for the evaluation window."""
    supplement = pd.read_csv(
        SUPPLEMENT, usecols=["PERMNO", "formation_month", "ret_final"], dtype={"formation_month": str}
    )
    cnn = pd.read_csv(
        CNN,
        usecols=["PERMNO", "formation_month", "holding_month", "cnn_decile", "formation_market_cap"],
        dtype={"formation_month": str, "holding_month": str},
    ).rename(columns={"cnn_decile": "decile", "formation_market_cap": "cap"})
    cnn = cnn.merge(supplement, on=["PERMNO", "formation_month"], how="left", validate="one_to_one")
    if start:
        cnn = cnn.loc[cnn["holding_month"] >= start]
    if end:
        cnn = cnn.loc[cnn["holding_month"] <= end]

    ff5 = load_french_monthly(FACTOR_DIR / "F-F_Research_Data_5_Factors_2x3_CSV.zip")
    st_rev = load_french_monthly(FACTOR_DIR / "F-F_ST_Reversal_Factor_CSV.zip")
    factors = ff5.join(st_rev, how="inner")

    rows = []
    for weighting, weighted in [("EW", False), ("VW", True)]:
        wide = monthly_portfolios(cnn, "ret_final", weighted)
        spread = (wide[10] - wide[1]).dropna()
        x_all = factors.reindex(spread.index)
        if x_all.isna().any().any():
            raise ValueError("Factor months are missing for the CNN holding period.")
        rows.append(
            {
                "weighting": weighting,
                "model": "raw spread",
                "months": int(len(spread)),
                "alpha_annual": 12 * spread.mean(),
                "alpha_t": newey_west_t_stat(spread, lag=4),
                "st_rev_loading": np.nan,
                "r_squared": np.nan,
            }
        )
        for model, columns in MODELS.items():
            x = x_all[columns]
            fit = ols_alpha_hac(spread.reset_index(drop=True), x.reset_index(drop=True), lag=4)
            design = np.column_stack([np.ones(len(x)), x.to_numpy()])
            coef, *_ = np.linalg.lstsq(design, spread.to_numpy(), rcond=None)
            residual = spread.to_numpy() - design @ coef
            rows.append(
                {
                    "weighting": weighting,
                    "model": model,
                    "months": int(len(spread)),
                    "alpha_annual": fit["annualized_alpha"],
                    "alpha_t": fit["alpha_t_stat_newey_west"],
                    "st_rev_loading": float(coef[-1]) if "st_rev" in columns else np.nan,
                    "r_squared": float(1 - residual.var() / spread.to_numpy().var()),
                }
            )
        rows[-1]["corr_with_st_rev"] = float(spread.corr(x_all["st_rev"]))

    result = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "".join(f"_{x}" for x in [start, end] if x)
    result.to_csv(OUTPUT_DIR / f"cnn_reversal_regressions{suffix}.csv", index=False)
    pd.set_option("display.width", 200)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
