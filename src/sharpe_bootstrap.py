"""Paired moving-block bootstrap of Sharpe-ratio differences against CNN only (D10 EW, net of 0.1%).

Input: ``outputs/cnn_plus_factor_subsets/d10_ew_net_monthly.csv`` (run ``cnn_plus_factor_subsets`` first).
Sharpe = sqrt(12) x mean / sd of returns in excess of RF. The same resampled months are used for every
design, so each difference keeps the cross-correlation. 5,000 draws, 95% percentile intervals for 6- and
12-month blocks. In-sample and exploratory: no multiple-comparison adjustment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402

FOLDER = ROOT / "outputs/cnn_plus_factor_subsets"
DRAWS, SEED, BLOCKS = 5000, 20260929, (6, 12)


def sharpe(excess: np.ndarray) -> np.ndarray:
    return np.sqrt(12) * excess.mean(axis=-2) / excess.std(axis=-2, ddof=1)


def block_indices(months: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, months - block + 1, size=(DRAWS, -(-months // block)))
    return (starts[..., None] + np.arange(block)).reshape(DRAWS, -1)[:, :months]


def paired_sharpe_difference(a: pd.Series, b: pd.Series, rf: pd.Series) -> dict:
    """Sharpe(a) - Sharpe(b) on returns in excess of RF, with 95% moving-block intervals on paired months."""
    excess = pd.concat([a, b], axis=1).sub(rf.reindex(a.index), axis=0).to_numpy()
    rng = np.random.default_rng(SEED)
    out = {"sharpe_diff": float(np.subtract(*sharpe(excess)))}
    for block in BLOCKS:
        draws = sharpe(excess[block_indices(len(excess), block, rng)])
        out[f"ci95_low_{block}m"], out[f"ci95_high_{block}m"] = np.percentile(draws[:, 0] - draws[:, 1], [2.5, 97.5])
    return out


def main() -> None:
    net = pd.read_csv(FOLDER / "d10_ew_net_monthly.csv", index_col=0).drop(columns="S&P 500 TR")
    net.index = net.index.astype(str)
    rf = load_french_monthly(FF5_MONTHLY_ZIP)["rf"]
    rf.index = rf.index.astype(str)
    excess = net.sub(rf.reindex(net.index), axis=0).to_numpy()
    base = list(net.columns).index("CNN only")

    observed = sharpe(excess)
    saved = pd.read_csv(FOLDER / "cnn_plus_factor_subsets.csv").query("portfolio == 'D10 EW'").set_index("design")["sharpe"]
    assert np.allclose(observed, saved.reindex(net.columns)), "Sharpe differs from cnn_plus_factor_subsets.csv"

    result = pd.DataFrame({"sharpe": observed, "diff_vs_cnn_only": observed - observed[base]}, index=net.columns)
    rng = np.random.default_rng(SEED)
    for block in BLOCKS:
        draws = sharpe(excess[block_indices(len(excess), block, rng)])
        diffs = draws - draws[:, [base]]
        result[f"ci95_low_{block}m"], result[f"ci95_high_{block}m"] = np.percentile(diffs, [2.5, 97.5], axis=0)
        result[f"share_above_0_{block}m"] = (diffs > 0).mean(axis=0)
    result.index.name = "design"
    result.to_csv(FOLDER / "sharpe_difference_bootstrap.csv")
    pd.set_option("display.width", 200)
    print(f"{len(excess)} months, {DRAWS} draws per block length")
    print(result.round(3).to_string())


if __name__ == "__main__":
    main()
