"""Re-evaluate IVOL and CNN deciles with the shared delisting-inclusive return.

Deciles are taken exactly as saved; only the realized t+1 return changes from the
professor ADJ_PRC return to ``ret_final`` from
``handoff/delisting_supplement``. Only portfolio-level aggregates are written.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "handoff" / "ivol_combination" / "src"))
from performance_metrics import newey_west_t_stat  # noqa: E402

SUPPLEMENT = ROOT / "handoff/delisting_supplement/next_month_returns_with_delisting_2000-2020.csv.gz"
IVOL = ROOT / "derived/professor_universe_2000_2020/ivol_deciles_adj_prc_returns_2000-2020.csv.gz"
CNN = ROOT / "outputs/cnn_oos_deciles_2001_2019/cnn_assignments_with_outcomes.csv.gz"
CRSP_MONTHLY = ROOT / "handoff/금공개 팀플/IVOL/CRSP_Monthly_1999-12_2020.csv.gz"
OUTPUT_DIR = ROOT / "outputs/delisting_robustness"
# CRSP MthDelFlg whose MthRet omits (G, M) or only partly includes (V) the delisting payoff.
INCOMPLETE_DELISTING = ("G", "M", "V")
KEY = ["PERMNO", "formation_month"]


@lru_cache(maxsize=1)
def formation_delisting_flags() -> pd.DataFrame:
    crsp = pd.read_csv(CRSP_MONTHLY, usecols=["PERMNO", "MthCalDt", "MthDelFlg"], keep_default_na=False, na_values=[""])
    crsp["formation_month"] = crsp["MthCalDt"].str[:7]
    crsp = crsp[[*KEY, "MthDelFlg"]].drop_duplicates()
    if crsp.duplicated(KEY).any():
        raise ValueError("Conflicting CRSP MthDelFlg values within one PERMNO-month.")
    return crsp


def tradable_at_formation(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop stock-months already delisting in the formation month (CRSP MthDelFlg not N).

    Such a stock cannot be bought at the month-end close, so it is removed before ranking.
    A month without a CRSP row is kept.
    """
    flags = frame[KEY].merge(formation_delisting_flags(), on=KEY, how="left", validate="many_to_one")["MthDelFlg"]
    return frame.loc[(flags.isna() | flags.eq("N")).to_numpy()].copy()


def load_supplement(columns: list[str], incomplete_delisting_return: float | None = None) -> pd.DataFrame:
    """Shared ``ret_final``. A number replaces the G/M/V delisting-month returns (stress test only)."""
    extra = ["ret_source", "crsp_mthdelflg"] if incomplete_delisting_return is not None else []
    frame = pd.read_csv(SUPPLEMENT, usecols=list(dict.fromkeys(columns + extra)),
                        dtype={"formation_month": str, "holding_month": str, "crsp_mthdelflg": str})
    if extra:
        incomplete = frame["ret_source"].eq("crsp_delisting") & frame["crsp_mthdelflg"].isin(INCOMPLETE_DELISTING)
        frame.loc[incomplete, "ret_final"] = incomplete_delisting_return
    return frame[columns]


def cnn_universe(incomplete_delisting_return: float | None = None) -> pd.DataFrame:
    """CNN-scored stock-months tradable at formation, deciles re-assigned after that filter, with ``ret_final``."""
    from common import assign_deciles

    frame = pd.read_csv(CNN, usecols=[*KEY, "holding_month", "cnn_prob_up", "formation_market_cap"],
                        dtype={"formation_month": str, "holding_month": str})
    frame = tradable_at_formation(frame.rename(columns={"formation_market_cap": "cap"}))
    frame["cnn_decile"] = assign_deciles(frame, signal_column="cnn_prob_up").astype(int)
    returns = load_supplement([*KEY, "ret_final"], incomplete_delisting_return)
    return frame.merge(returns, on=KEY, how="left", validate="one_to_one")


def monthly_portfolios(frame: pd.DataFrame, return_column: str, weighted: bool) -> pd.DataFrame:
    usable = frame.dropna(subset=[return_column]).copy()
    if weighted:
        usable = usable.loc[usable["cap"].gt(0)].copy()
        usable["weighted"] = usable[return_column] * usable["cap"]
        grouped = usable.groupby(["holding_month", "decile"]).agg(
            weighted=("weighted", "sum"), weight=("cap", "sum")
        )
        series = grouped["weighted"] / grouped["weight"]
    else:
        series = usable.groupby(["holding_month", "decile"])[return_column].mean()
    return series.unstack("decile").sort_index()


def evaluate(frame: pd.DataFrame, long: int, short: int, label: str) -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    decile_rows: list[dict] = []
    for weighting, weighted in [("EW", False), ("VW", True)]:
        for version, column in [("adj_prc_only", "ret_adjprc_eval"), ("with_delisting", "ret_final")]:
            wide = monthly_portfolios(frame, column, weighted)
            spread = (wide[long] - wide[short]).dropna()
            summary_rows.append(
                {
                    "strategy": label,
                    "weighting": weighting,
                    "version": version,
                    "spread": f"Q{long}-Q{short}",
                    "months": int(len(spread)),
                    "annualized_mean": float(12 * spread.mean()),
                    "annualized_volatility": float(np.sqrt(12) * spread.std(ddof=1)),
                    "newey_west_t_lag4": float(newey_west_t_stat(spread, lag=4)),
                }
            )
            for decile in range(1, 11):
                decile_rows.append(
                    {
                        "strategy": label,
                        "weighting": weighting,
                        "version": version,
                        "decile": decile,
                        "annualized_mean": float(12 * wide[decile].mean()),
                        "annualized_volatility": float(np.sqrt(12) * wide[decile].std(ddof=1)),
                    }
                )
    return summary_rows, decile_rows


def coverage(frame: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for decile, group in frame.groupby("decile"):
        rows.append(
            {
                "strategy": label,
                "decile": int(decile),
                "rows": int(len(group)),
                "missing_before": int(group["ret_adjprc_eval"].isna().sum()),
                "missing_after": int(group["ret_final"].isna().sum()),
                "replaced_by_crsp_delisting": int(group["ret_source"].eq("crsp_delisting").sum()),
                "filled_by_crsp_fill": int(group["ret_source"].eq("crsp_fill").sum()),
                "mean_crsp_delisting_return": float(
                    group.loc[group["ret_source"].eq("crsp_delisting"), "ret_final"].mean()
                ),
            }
        )
    return rows


def main() -> None:
    supplement = pd.read_csv(
        SUPPLEMENT,
        usecols=["PERMNO", "formation_month", "ret_adjprc", "ret_final", "ret_source"],
        dtype={"formation_month": str},
    )

    ivol = pd.read_csv(
        IVOL,
        usecols=["PERMNO", "formation_month", "decile", "next_month", "next_month_return", "formation_market_cap"],
        dtype={"formation_month": str, "next_month": str},
    ).rename(columns={"next_month": "holding_month", "next_month_return": "ret_adjprc_eval", "formation_market_cap": "cap"})
    cnn = pd.read_csv(
        CNN,
        usecols=["PERMNO", "formation_month", "holding_month", "cnn_decile", "next_month_return", "formation_market_cap"],
        dtype={"formation_month": str, "holding_month": str},
    ).rename(columns={"cnn_decile": "decile", "next_month_return": "ret_adjprc_eval", "formation_market_cap": "cap"})

    checks = {}
    results_summary: list[dict] = []
    results_deciles: list[dict] = []
    results_coverage: list[dict] = []
    for label, frame, long, short in [("IVOL", ivol, 1, 10), ("CNN", cnn, 10, 1)]:
        merged = frame.merge(
            supplement, on=["PERMNO", "formation_month"], how="left", validate="one_to_one"
        )
        both = merged["ret_adjprc_eval"].notna() & merged["ret_adjprc"].notna()
        checks[label] = {
            "rows": int(len(merged)),
            "rows_not_in_supplement": int(merged["ret_source"].isna().sum()),
            "saved_return_matches_supplement_adj_prc": bool(
                np.allclose(merged.loc[both, "ret_adjprc_eval"], merged.loc[both, "ret_adjprc"], rtol=0, atol=1e-12)
            ),
            "saved_missing_equals_supplement_missing": bool(
                merged["ret_adjprc_eval"].isna().equals(merged["ret_adjprc"].isna())
            ),
        }
        summary, deciles = evaluate(merged, long, short, label)
        results_summary += summary
        results_deciles += deciles
        results_coverage += coverage(merged, label)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_summary).to_csv(OUTPUT_DIR / "spread_summary.csv", index=False)
    pd.DataFrame(results_deciles).to_csv(OUTPUT_DIR / "decile_summary.csv", index=False)
    pd.DataFrame(results_coverage).to_csv(OUTPUT_DIR / "coverage_by_decile.csv", index=False)
    (OUTPUT_DIR / "checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(checks, indent=2))
    print(pd.DataFrame(results_summary).round(4).to_string(index=False))
    print(pd.DataFrame(results_coverage).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
