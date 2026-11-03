from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def newey_west_t_stat(values: pd.Series, lag: int = 4) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    n_obs = len(clean)
    if n_obs < 2 or lag < 0:
        return math.nan
    demeaned = clean - clean.mean()
    long_run_variance = float(demeaned @ demeaned / n_obs)
    for offset in range(1, min(lag, n_obs - 1) + 1):
        autocovariance = float(demeaned[offset:] @ demeaned[:-offset] / n_obs)
        weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * weight * autocovariance
    if long_run_variance <= 0:
        return math.nan
    return float(clean.mean() / math.sqrt(long_run_variance / n_obs))


def maximum_drawdown(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(clean) == 0:
        return math.nan
    wealth = np.cumprod(1.0 + clean)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))
    path = np.concatenate(([1.0], wealth))
    return float(np.min(path / peaks - 1.0))


def summarize_return_series(
    returns: pd.Series,
    *,
    risk_free: pd.Series | None = None,
    return_kind: str = "total",
    newey_west_lag: int = 4,
) -> dict[str, int | float | str]:
    """Summarize one monthly return series under the shared project convention."""

    if return_kind not in {"total", "excess"}:
        raise ValueError("return_kind must be 'total' or 'excess'")
    frame = pd.DataFrame({"return": pd.to_numeric(returns, errors="coerce")})
    if risk_free is not None:
        frame["rf"] = pd.to_numeric(risk_free, errors="coerce")
    frame = frame.dropna(subset=["return"])
    if frame.empty:
        raise ValueError("Return series has no usable observations")
    if return_kind == "total":
        if risk_free is None:
            excess = None
            sharpe_basis = "unavailable because no risk-free series was supplied"
        else:
            frame = frame.dropna(subset=["rf"])
            excess = frame["return"] - frame["rf"]
            sharpe_basis = "return minus supplied monthly risk-free return"
    else:
        excess = frame["return"]
        sharpe_basis = "input return is already excess/self-financing"
    n_obs = len(frame)
    mean_return = float(frame["return"].mean())
    mean_excess = float(excess.mean()) if excess is not None else math.nan
    monthly_std = float(frame["return"].std(ddof=1)) if n_obs > 1 else math.nan
    iid_t = (
        mean_return / (monthly_std / math.sqrt(n_obs))
        if n_obs > 1 and monthly_std > 0
        else math.nan
    )
    terminal_wealth = float(np.prod(1.0 + frame["return"].to_numpy(float)))
    cumulative = terminal_wealth - 1.0
    cagr = terminal_wealth ** (12.0 / n_obs) - 1.0 if terminal_wealth > 0 else math.nan
    annualized_volatility = math.sqrt(12.0) * monthly_std
    annualized_sharpe_like = (
        math.sqrt(12.0) * mean_return / monthly_std if monthly_std > 0 else math.nan
    )
    excess_std = (
        float(excess.std(ddof=1))
        if excess is not None and len(excess) > 1
        else math.nan
    )
    annualized_sharpe = (
        math.sqrt(12.0) * mean_excess / excess_std
        if excess is not None and excess_std > 0
        else math.nan
    )
    return {
        "n_months": int(n_obs),
        "mean_monthly_return": mean_return,
        "mean_monthly_excess_return": mean_excess,
        "annualized_mean_return": 12.0 * mean_return,
        "annualized_volatility": annualized_volatility,
        "annualized_sharpe": annualized_sharpe,
        "annualized_sharpe_like": annualized_sharpe_like,
        "sharpe_basis": sharpe_basis,
        "cumulative_return": cumulative,
        "cagr": cagr,
        "maximum_drawdown": maximum_drawdown(frame["return"]),
        "t_stat_mean_return_iid": iid_t,
        f"t_stat_mean_return_newey_west_lag{newey_west_lag}": newey_west_t_stat(
            frame["return"], lag=newey_west_lag
        ),
    }


def summarize_portfolios(
    monthly: pd.DataFrame,
    *,
    date_column: str = "holding_month",
    portfolio_column: str = "portfolio",
    return_column: str = "return",
    return_kind: str = "total",
) -> pd.DataFrame:
    required = {date_column, portfolio_column, return_column}
    missing = sorted(required.difference(monthly.columns))
    if missing:
        raise ValueError(f"Monthly portfolio table is missing columns: {missing}")
    if monthly.duplicated([date_column, portfolio_column]).any():
        raise ValueError("Monthly portfolio table has duplicate date/portfolio keys")
    records: list[dict[str, Any]] = []
    for portfolio, group in monthly.groupby(portfolio_column, observed=True, sort=True):
        summary = summarize_return_series(
            group.sort_values(date_column)[return_column], return_kind=return_kind
        )
        records.append({"portfolio": str(portfolio), **summary})
    return pd.DataFrame.from_records(records)


def add_long_short(
    monthly: pd.DataFrame,
    *,
    long_label: str,
    short_label: str,
    spread_label: str,
    date_column: str = "holding_month",
) -> pd.DataFrame:
    if monthly.duplicated([date_column, "portfolio"]).any():
        raise ValueError("Monthly table has duplicate date/portfolio keys")
    wide = monthly.pivot(index=date_column, columns="portfolio", values="return")
    missing = sorted({long_label, short_label}.difference(wide.columns))
    if missing:
        raise ValueError(f"Cannot build spread; missing portfolios: {missing}")
    spread = (wide[long_label] - wide[short_label]).rename("return").reset_index()
    spread["portfolio"] = spread_label
    return pd.concat(
        [monthly[[date_column, "portfolio", "return"]], spread], ignore_index=True
    )


def ols_alpha_hac(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    lag: int = 4,
) -> dict[str, float | int]:
    """Monthly OLS alpha and Newey-West t-stat with a Bartlett kernel."""

    y = pd.to_numeric(returns, errors="coerce")
    x = factors.apply(pd.to_numeric, errors="coerce")
    joined = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(joined) <= len(x.columns) + 1:
        raise ValueError("Insufficient complete observations for factor regression")
    yv = joined.pop("y").to_numpy(float)
    xv = np.column_stack([np.ones(len(joined)), joined.to_numpy(float)])
    rank = int(np.linalg.matrix_rank(xv))
    if rank != xv.shape[1]:
        raise ValueError("Factor regression design matrix is singular")
    beta = np.linalg.lstsq(xv, yv, rcond=None)[0]
    residual = yv - xv @ beta
    bread = np.linalg.inv(xv.T @ xv)
    meat = np.zeros((xv.shape[1], xv.shape[1]), dtype=float)
    for t in range(len(xv)):
        score = xv[t] * residual[t]
        meat += np.outer(score, score)
    for offset in range(1, min(lag, len(xv) - 1) + 1):
        weight = 1.0 - offset / (lag + 1.0)
        cross = np.zeros_like(meat)
        for t in range(offset, len(xv)):
            cross += np.outer(xv[t] * residual[t], xv[t - offset] * residual[t - offset])
        meat += weight * (cross + cross.T)
    covariance = bread @ meat @ bread
    alpha_se = math.sqrt(max(float(covariance[0, 0]), 0.0))
    return {
        "n_months": int(len(xv)),
        "monthly_alpha": float(beta[0]),
        "annualized_alpha": float(12.0 * beta[0]),
        "alpha_t_stat_newey_west": float(beta[0] / alpha_se)
        if alpha_se > 0
        else math.nan,
        "regression_rank": rank,
    }
