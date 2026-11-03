"""Signal-combination methods other than rank averaging (professor meeting, 2026-09-30).

Pure functions on a stock-month frame; no file access. Used by ``alt_combination.py`` (team data, four
signals with CNN) and by ``pilot/pilot_three_signal.py`` (CRSP-daily reconstruction, three signals).

Pre-specified on 2026-10-02, before any result was seen. Every design is reported whatever its result.

A signal is a column plus a direction: +1 if a higher value is the preferred side (momentum, CNN P(up)),
-1 if a lower value is preferred (FP beta, IVOL). All transforms are done month by month (formation month).

RANK  (current team method, for reference)
      rank percentile in [0, 1] per signal, flipped for direction -1, weighted sum.
Z     winsorized z-score per signal: clip at the month's 1st and 99th percentiles, subtract the mean,
      divide by the standard deviation (ddof=1), multiply by the direction, weighted sum.
      Same weight sets as RANK, so RANK -> Z is the only change ("rank -> size of the value").
RA    risk-adjusted CNN: (CNN P(up) - month median) / IVOL, IVOL clipped at the month's 1st/99th
      percentiles first. CNN is the return engine; IVOL is the risk it is scaled by (alpha / residual
      risk, Grinold-Kahn). Momentum and beta are not used.
FM    Fama-MacBeth expected return. Each month s: OLS of the next-month return on an intercept and the
      Z-transformed signals (EW, all stocks with a return). At formation month t the slopes of months
      s = t-60 .. t-1 are averaged (at least 24 months), and the score is z_i,t . mean slope.
      Month s uses the return of s+1, which is known at the end of t, so there is no look-ahead.
      Slopes are unconstrained (a signal with a negative average slope gets a negative weight).
Scores are rounded to 12 decimals, as in the team's ``combined_score``, so ties are reproducible.

Added 2026-10-02 after the three-signal pilot, before it was run anywhere (exploratory, reported regardless):
ZLOG  as Z, but after winsorizing, momentum becomes log(1 + mom) and FP beta and IVOL become log(value)
      (CNN unchanged), then standardize. The pilot showed that plain Z lets momentum's long right tail
      dominate while the preferred (low) side of IVOL and beta is squeezed near zero; logs make the
      distributions closer to symmetric. ZLOG tests whether Z failed because of tails or because of values.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MONTH = "formation_month"
FM_WINDOW, FM_MIN_MONTHS = 60, 24
WINSOR = (0.01, 0.99)


# ----------------------------------------------------------------------------------------------- transforms
def pct_rank(values: pd.Series) -> pd.Series:
    """Rank percentile in [0, 1] (average ties), identical to the team's ``pct_rank``."""
    return (values.rank(method="average") - 1.0) / (len(values) - 1.0)


def winsorize(values: pd.Series, lower: float = WINSOR[0], upper: float = WINSOR[1]) -> pd.Series:
    lo, hi = values.quantile([lower, upper])
    return values.clip(lo, hi)


def zscore(values: pd.Series) -> pd.Series:
    w = winsorize(values)
    sd = w.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index)
    return (w - w.mean()) / sd


LOG_TRANSFORM = {"mom": np.log1p, "beta": np.log, "ivol_daily": np.log, "ivol": np.log}


def zscore_log(values: pd.Series, column: str) -> pd.Series:
    """Winsorize, apply the column's log transform (if any), then standardize."""
    w = winsorize(values)
    f = LOG_TRANSFORM.get(column)
    x = pd.Series(f(w.to_numpy(dtype=float)), index=w.index) if f else w
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index)
    return (x - x.mean()) / sd


def transformed(frame: pd.DataFrame, signals: list[tuple[str, int]], how: str) -> pd.DataFrame:
    """One column per signal, higher = preferred. ``how`` is 'rank', 'z' or 'zlog'."""
    g = frame.groupby(MONTH)
    out = {}
    for column, direction in signals:
        if how == "rank":
            r = g[column].transform(pct_rank)
            out[column] = r if direction > 0 else 1.0 - r
        elif how == "z":
            out[column] = direction * g[column].transform(zscore)
        elif how == "zlog":
            out[column] = direction * g[column].transform(lambda v, c=column: zscore_log(v, c))
        else:
            raise ValueError(how)
    return pd.DataFrame(out, index=frame.index)


def weighted(columns: pd.DataFrame, weights) -> np.ndarray:
    """Weighted sum added column by column in the given order (bit-identical to ``combined_score``)."""
    total = np.zeros(len(columns))
    for column, weight in zip(columns.columns, weights):
        total = total + float(weight) * columns[column].to_numpy(dtype=float)
    return np.round(total, 12)


def rank_score(frame: pd.DataFrame, signals, weights) -> np.ndarray:
    return weighted(transformed(frame, signals, "rank"), weights)


def z_score(frame: pd.DataFrame, signals, weights) -> np.ndarray:
    return weighted(transformed(frame, signals, "z"), weights)


def zlog_score(frame: pd.DataFrame, signals, weights) -> np.ndarray:
    return weighted(transformed(frame, signals, "zlog"), weights)


def risk_adjusted_score(frame: pd.DataFrame, engine: str, risk: str) -> np.ndarray:
    g = frame.groupby(MONTH)
    centered = frame[engine] - g[engine].transform("median")
    scaled_risk = g[risk].transform(winsorize)
    return np.round((centered / scaled_risk).to_numpy(dtype=float), 12)


# ----------------------------------------------------------------------------------------------- Fama-MacBeth
def cross_section_slopes(frame: pd.DataFrame, signals, ret: str) -> pd.DataFrame:
    """Monthly OLS slopes of ``ret`` (next-month return) on the Z-transformed signals, by formation month."""
    z = transformed(frame, signals, "z")
    names = [c for c, _ in signals]
    rows = []
    for month, idx in frame.groupby(MONTH, sort=True).groups.items():
        y = frame.loc[idx, ret].to_numpy(dtype=float)
        x = z.loc[idx, names].to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if ok.sum() < len(names) + 10:
            continue
        design = np.column_stack([np.ones(ok.sum()), x[ok]])
        coef, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
        rows.append({MONTH: month, "n": int(ok.sum()), "intercept": coef[0], **dict(zip(names, coef[1:]))})
    return pd.DataFrame(rows).set_index(MONTH)


def trailing_lambdas(slopes: pd.DataFrame, months, window: int = FM_WINDOW, min_months: int = FM_MIN_MONTHS) -> pd.DataFrame:
    """Average slope available at the end of each formation month t: months s in [t-window, t-1]."""
    periods = pd.PeriodIndex(slopes.index, freq="M")
    names = [c for c in slopes.columns if c not in ("n", "intercept")]
    out = {}
    for month in months:
        t = pd.Period(month, freq="M")
        use = (periods <= t - 1) & (periods >= t - window)
        if use.sum() >= min_months:
            out[month] = {**slopes.loc[use, names].mean().to_dict(), "months_used": int(use.sum())}
    return pd.DataFrame.from_dict(out, orient="index")


def fama_macbeth_score(frame: pd.DataFrame, signals, ret: str, window: int = FM_WINDOW,
                       min_months: int = FM_MIN_MONTHS) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Expected-return score (NaN where fewer than ``min_months`` past slopes exist), slopes, lambdas."""
    slopes = cross_section_slopes(frame, signals, ret)
    months = sorted(frame[MONTH].unique())
    lam = trailing_lambdas(slopes, months, window, min_months)
    names = [c for c, _ in signals]
    z = transformed(frame, signals, "z")
    weights = lam.reindex(frame[MONTH].to_numpy())[names].to_numpy(dtype=float)
    score = pd.Series(np.round((z[names].to_numpy(dtype=float) * weights).sum(axis=1), 12), index=frame.index)
    score[np.isnan(weights).any(axis=1)] = np.nan
    return score, slopes, lam


# ----------------------------------------------------------------------------------------------- portfolios
def assign_deciles(frame: pd.DataFrame, signal_column: str, month_column: str = MONTH, id_column: str = "PERMNO") -> pd.Series:
    """Deterministic equal-count deciles, ties by PERMNO ascending (same as the team's ``common.assign_deciles``)."""
    result = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for _, index in frame.groupby(month_column, sort=True).groups.items():
        ordered = frame.loc[index].sort_values([signal_column, id_column], kind="mergesort")
        n_rows = len(ordered)
        if n_rows:
            result.loc[ordered.index] = (np.arange(n_rows, dtype=np.int64) * 10 // n_rows) + 1
    return result


def top_n_returns(frame: pd.DataFrame, score: str, n: int, weighting: str, ret: str = "ret_final",
                  cost: float = 0.001) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same logic as the team's ``top_n_returns`` (drift-adjusted turnover, cost per dollar traded)."""
    ranked = frame.sort_values([MONTH, score, "PERMNO"], ascending=[True, False, True])
    top = ranked.groupby(MONTH, sort=True).head(n).copy()
    if weighting == "VW":
        top = top.loc[top["cap"].gt(0)].copy()
        top["weight"] = top["cap"] / top.groupby(MONTH)["cap"].transform("sum")
    else:
        top["weight"] = 1.0 / top.groupby(MONTH)["PERMNO"].transform("size")
    records, previous = [], None
    for _, holdings in top.groupby(MONTH, sort=True):
        target = holdings.set_index("PERMNO")["weight"]
        returns = holdings.set_index("PERMNO")[ret]
        available = returns.notna()
        gross = float((target[available] * returns[available]).sum() / target[available].sum())
        if previous is None:
            traded = 1.0
        else:
            union = target.index.union(previous.index)
            traded = float((target.reindex(union, fill_value=0.0) - previous.reindex(union, fill_value=0.0)).abs().sum())
        drifted = target[available] * (1.0 + returns[available])
        previous = drifted / drifted.sum()
        records.append({"holding_month": holdings["holding_month"].iloc[0], "gross": gross, "traded": traded,
                        "net": gross - cost * traded, "holdings": int(len(holdings)),
                        "missing_returns": int((~available).sum())})
    return pd.DataFrame(records).set_index("holding_month"), top


# ----------------------------------------------------------------------------------------------- statistics
DRAWS, SEED, BLOCKS = 5000, 20260929, (6, 12)


def _sharpe(excess: np.ndarray) -> np.ndarray:
    return np.sqrt(12) * excess.mean(axis=-2) / excess.std(axis=-2, ddof=1)


def _block_indices(months: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, months - block + 1, size=(DRAWS, -(-months // block)))
    return (starts[..., None] + np.arange(block)).reshape(DRAWS, -1)[:, :months]


def paired_sharpe_difference(a: pd.Series, b: pd.Series, rf: pd.Series) -> dict:
    """Same as the team's ``sharpe_bootstrap.paired_sharpe_difference`` (common months only)."""
    both = pd.concat([a, b], axis=1).dropna()
    excess = both.sub(rf.reindex(both.index), axis=0).to_numpy()
    rng = np.random.default_rng(SEED)
    out = {"months": len(both), "sharpe_diff": float(np.subtract(*_sharpe(excess)))}
    for block in BLOCKS:
        draws = _sharpe(excess[_block_indices(len(excess), block, rng)])
        out[f"ci95_low_{block}m"], out[f"ci95_high_{block}m"] = np.percentile(draws[:, 0] - draws[:, 1], [2.5, 97.5])
    return out


def max_drawdown(r: pd.Series) -> float:
    wealth = np.cumprod(1.0 + r.to_numpy(dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float(np.min(wealth / peak - 1.0))


def basic_metrics(r: pd.Series, rf: pd.Series) -> dict:
    x = r - rf.reindex(r.index)
    return {"months": len(r), "annual_mean": 12 * r.mean(), "cagr": float(np.prod(1 + r)) ** (12 / len(r)) - 1,
            "volatility": math.sqrt(12) * r.std(ddof=1), "sharpe": math.sqrt(12) * x.mean() / x.std(ddof=1),
            "max_drawdown": max_drawdown(r)}


def newey_west_t(s: pd.Series, lag: int) -> float:
    e = (s - s.mean()).to_numpy(dtype=float)
    n = len(e)
    var = e @ e / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(s.mean() / math.sqrt(var / n))


def fm_summary(slopes: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Classic Fama-MacBeth table on formation months start..end: mean monthly slope, plain and NW(4) t."""
    part = slopes.loc[(slopes.index >= start) & (slopes.index <= end)]
    rows = []
    for c in [c for c in part.columns if c != "n"]:
        s = part[c].dropna()
        rows.append({"coefficient": c, "months": len(s), "mean_monthly": s.mean(),
                     "t_plain": s.mean() / (s.std(ddof=1) / math.sqrt(len(s))), "t_nw4": newey_west_t(s, 4),
                     "share_positive": float((s > 0).mean())})
    return pd.DataFrame(rows)
