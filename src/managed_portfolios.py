"""Barroso & Santa-Clara (2015) risk-managed momentum and Han, Huang & Zhou (2019) trend-managed holdings.

Usage: python src/managed_portfolios.py PATH/TO/sp500_monthly.csv [--covid]
(--covid: the same designs on the three-factor score, run on to December 2020, since CNN images end at
the 2019-12 formation month. 2020 is a stress check, not validation.)
Pre-specified on 2026-09-29 before running; every row below is reported.

A. BSC replication on our data. WML = momentum decile 10 minus decile 1 (P(t-1)/P(t-12)-1, professor
   universe, tradable at formation, equal-count deciles), EW and VW (formation-month cap). Daily returns of a
   portfolio are buy-and-hold within the month (``Daily.portfolio``), an input for the risk forecast built from
   the daily panel, not an exact replica of the paper's data. Variance forecast for holding month t: 21 x mean of the squared daily
   WML returns over the last 126 trading days before t (BSC eq. 5); scaled WML = target / sigma_hat x WML
   (eq. 6), target 12% a year. The daily panel has no delisting days, so the part of each delisting (or
   CRSP-filled) month's ``ret_final`` missing from it is booked on the month's last market day.
   HHZ is not replicated: its eight anomalies need annual Compustat data.
B. Application to the product: Equal 25% score (CNN, MOM, low beta, low IVOL ranks) on the four-signal
   universe; long-only decile 10 and top 50, EW, net of 0.1% per dollar traded.
     Base
     BSC    portfolio volatility (called BSC-B before 2026-09-30): hold e_t = min(1, target / sigma_hat_p) of the
            portfolio, the rest in T-bills; sigma_hat_p is eq. 5 on the unscaled portfolio's daily returns (the
            paper's estimator applied to the whole long-only portfolio; ``leverage_bsc.py`` lifts the cap)
     HHZ    trend, cash: a selected stock is held only in a confirmed uptrend, MA50 > MA200 of its daily
            total-return index over the last 50 / 200 market days to the formation month's last market day, each
            window with at least 80% valid days (the teammate package's rule, adopted 2026-09-30). Otherwise (down,
            equal, or unknown) its weight earns the T-bill rate (HHZ long-leg rule)
     HHZ-R  trend, reweighted: the same, but the weight goes to the kept stocks (sensitivity)
     BSC + HHZ (HHZ in cash form; BSC scales the HHZ portfolio by its own forecast)
     DM     appendix only, not for the professor meeting (called BSC-A before): the MOM rank weight is
            0.25 x k_t, k_t = target / sigma_hat of the EW WML, read with Daniel & Moskowitz (2016); DM fixed k
            (diagnostic, k at its 2005-2019 average, ex post) and DM + HHZ
   A forecast needs at least 21 daily returns; before that the portfolio is unscaled. Months run on an explicit calendar: a month with no
   holdings is all cash. Every design is compared with Base by a paired block bootstrap (exploratory, no
   multiple-test adjustment), over four 45-month subperiods, and saved month by month.
Holding months 2005-01 to 2019-12 (``eval_window``). Only portfolio-level aggregates are written.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fp_bab  # noqa: E402
from common import assign_deciles  # noqa: E402
from delisting_robustness import SUPPLEMENT, load_supplement, monthly_portfolios, tradable_at_formation  # noqa: E402
from eval_window import END, FIRST_FORMATION, START  # noqa: E402
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly  # noqa: E402
from longonly_d10_vs_sp500 import metrics  # noqa: E402
from momentum_lowbeta_deciles import CRSP_MONTHLY, load_prices, momentum_signals  # noqa: E402
from performance_metrics import maximum_drawdown  # noqa: E402
from sharpe_bootstrap import paired_sharpe_difference  # noqa: E402
from professor_universe import load_formation_caps  # noqa: E402
from three_factor_combination import build_panel, combined_score, pct_rank  # noqa: E402
from top_n_combination import CNN_SCORES  # noqa: E402

OUTPUT_DIR = ROOT / "outputs/managed_portfolios"
TARGET = 0.12 / math.sqrt(12)  # monthly target volatility
WINDOW, MIN_DAYS = 126, 21
COST = 0.001
KEY = ["PERMNO", "formation_month"]


def month_code(months: pd.Series) -> np.ndarray:
    return (months.str[:4].astype(int) * 12 + months.str[5:7].astype(int) - 1).to_numpy()


def code_month(codes) -> list[str]:
    return [f"{c // 12}-{c % 12 + 1:02d}" for c in codes]


class Daily:
    """Daily stock returns of the professor universe, with month codes and the daily T-bill rate."""

    def __init__(self) -> None:
        market = fp_bab.load_market()
        self.day_month = (market["date"].dt.year * 12 + market["date"].dt.month - 1).to_numpy()
        self.rf = market["rf"].to_numpy()
        days = fp_bab.load_stock_days(market)
        days["mcode"] = self.day_month[days["day"].to_numpy()]
        days["r"] = np.expm1(days["l"].to_numpy())
        self.days = days
        self.residuals = self.missing_monthly_returns()

    def missing_monthly_returns(self) -> pd.DataFrame:
        """Delisting months and CRSP-filled months (``ret_source`` crsp_delisting / crsp_fill) have no or only
        partial rows in the daily panel, which has no delisting days. The gap between ``ret_final`` and the daily
        returns is booked on the month's last market day, so the daily returns compound to the monthly return."""
        sup = pd.read_csv(SUPPLEMENT, usecols=["PERMNO", "holding_month", "ret_final", "ret_source"], dtype={"holding_month": str})
        sup = sup.loc[sup["ret_source"].isin(["crsp_delisting", "crsp_fill"]) & sup["ret_final"].notna()]
        sup = sup.assign(mcode=month_code(sup["holding_month"]))
        seen = self.days.merge(sup[["PERMNO", "mcode"]], on=["PERMNO", "mcode"]).groupby(["PERMNO", "mcode"])["l"].sum()
        sup = sup.merge(seen.reset_index(), on=["PERMNO", "mcode"], how="left")
        days = pd.Series(np.arange(len(self.day_month))).groupby(self.day_month)
        out = pd.DataFrame({"PERMNO": sup["PERMNO"].to_numpy(), "mcode": sup["mcode"].to_numpy(),
                            "day_end": days.max().reindex(sup["mcode"]).to_numpy(),
                            "day_start": days.min().reindex(sup["mcode"]).to_numpy(),
                            "r": ((1 + sup["ret_final"]) / np.exp(sup["l"].fillna(0.0)) - 1).to_numpy()})
        return out.loc[(out["r"].abs() > 1e-12) & out["day_end"].notna()].astype({"day_end": int, "day_start": int})

    def portfolio(self, members: pd.DataFrame, rebalance_daily: bool = False, residual_day: str = "day_end") -> pd.Series:
        """Daily return of holdings (PERMNO, mcode = holding month, w; weights sum to <= 1, the rest earns the
        T-bill rate), including the monthly returns missing from the daily panel (``missing_monthly_returns``,
        booked on ``residual_day``: "day_end" or "day_start" of the month).
        Default: buy-and-hold within the month, as the portfolio is traded (weights drift with returns; a stock
        without a return that day keeps its value). ``rebalance_daily`` resets the formation weights every day
        over the stocks trading that day (the round-2 approximation, kept for comparison)."""
        keys = ["PERMNO", "mcode"]
        residuals = self.residuals.rename(columns={residual_day: "day"})
        d = pd.concat([self.days.merge(members[keys], on=keys)[[*keys, "day", "r"]],
                       residuals.merge(members[keys], on=keys)[[*keys, "day", "r"]]], ignore_index=True)
        with np.errstate(divide="ignore"):
            d["l"] = np.log1p(d["r"])
        d = d.groupby([*keys, "day"], as_index=False)["l"].sum()
        d["r"] = np.expm1(d["l"])
        if rebalance_daily:
            d = d.merge(members, on=keys)
            d["wr"] = d["w"] * d["r"]
            g = d.groupby("day")[["wr", "w"]].sum()
            share = members.groupby("mcode")["w"].sum().reindex(self.day_month[g.index]).to_numpy()
            return pd.Series(g["wr"].to_numpy() / g["w"].to_numpy() * share + (1 - share) * self.rf[g.index], index=g.index)
        by_month, out = dict(tuple(d.groupby("mcode"))), []
        for mcode, held in members.groupby("mcode", sort=True):
            days = np.flatnonzero(self.day_month == mcode)
            weights = held.set_index("PERMNO")["w"]
            part = by_month.get(mcode)
            r = part.pivot(index="day", columns="PERMNO", values="r") if part is not None else pd.DataFrame()
            r = r.reindex(index=days, columns=weights.index).fillna(0.0).to_numpy()
            value = np.cumprod(1.0 + r, axis=0) @ weights.to_numpy() + (1.0 - weights.sum()) * np.cumprod(1.0 + self.rf[days])
            out.append(pd.Series(value / np.r_[1.0, value[:-1]] - 1.0, index=days))
        return pd.concat(out)

    def forecast(self, daily: pd.Series) -> pd.Series:
        """BSC eq. 5: monthly volatility forecast for each holding month code, from the days before it."""
        variance = daily.pow(2).rolling(WINDOW, min_periods=MIN_DAYS).mean() * 21
        at_month_end = variance.groupby(self.day_month[daily.index]).last()
        at_month_end.index = at_month_end.index + 1
        return np.sqrt(at_month_end)

    def trend(self, short: int = 50, long: int = 200, min_valid: float = 0.8) -> pd.DataFrame:
        """HHZ confirmed-uptrend test as in the teammate package (adopted 2026-09-30): moving averages of each stock's
        daily total-return index over the last ``short`` / ``long`` market days to the formation month's last market
        day, each the mean over the stock's valid days in the window and requiring at least ``min_valid`` of them.
        no_uptrend = not (MA(short) > MA(long)): a falling or flat trend, or an unknown one (too few valid days)."""
        perm, day, l, mcode = (self.days[c].to_numpy() for c in ["PERMNO", "day", "l", "mcode"])
        new = np.r_[True, perm[1:] != perm[:-1]]
        start, group = np.flatnonzero(new), np.cumsum(new) - 1
        total = np.cumsum(l)
        level = np.exp(total - (total[start] - l[start])[group])  # total-return index within each stock
        csum = np.r_[0.0, np.cumsum(level)]
        span = len(self.day_month) + 1
        key = group * span + day
        if not (np.diff(key) > 0).all():
            raise ValueError("Daily rows must be sorted by PERMNO and day without duplicates.")
        last = np.r_[(perm[1:] != perm[:-1]) | (mcode[1:] != mcode[:-1]), True]  # stock traded in the month
        month_end = pd.Series(np.arange(len(self.day_month))).groupby(self.day_month).max()
        end = month_end.reindex(mcode[last]).to_numpy()
        base = group[last] * span

        def moving_average(n: int) -> np.ndarray:
            lo = np.searchsorted(key, base + np.maximum(end - n + 1, 0))
            hi = np.searchsorted(key, base + end, side="right")
            valid = hi - lo
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(valid >= math.ceil(min_valid * n), (csum[hi] - csum[lo]) / valid, np.nan)

        out = pd.DataFrame({"PERMNO": perm[last], "formation_month": code_month(mcode[last])})
        out["no_uptrend"] = ~(moving_average(short) > moving_average(long))  # NaN (unknown) compares False
        return out[[*KEY, "no_uptrend"]]


def managed_returns(holdings: pd.DataFrame, rf: pd.Series, months: list[str] | None = None) -> pd.DataFrame:
    """Monthly gross/net returns of target weights w (sum <= 1, rest in T-bills), drift-adjusted turnover.

    ``months`` is the full calendar: a month without holdings is all cash (earns RF, sells everything held)."""
    groups = dict(tuple(holdings.groupby("holding_month", sort=True)))
    months = sorted(groups) if months is None else months
    records, previous = [], pd.Series(dtype=float)
    for month in months:
        group = groups.get(month, holdings.iloc[:0])
        target = group.set_index("PERMNO")["w"]
        returns = group.set_index("PERMNO")["ret_final"]
        available = returns.notna()
        invested = float(target.sum())
        weights = target[available] * (invested / target[available].sum()) if available.any() else target[available]
        cash = 1.0 - float(weights.sum())
        gross = float((weights * returns[available]).sum()) + cash * float(rf[month])
        union = target.index.union(previous.index)
        traded = float((target.reindex(union, fill_value=0.0) - previous.reindex(union, fill_value=0.0)).abs().sum())
        previous = weights * (1.0 + returns[available]) / (1.0 + gross)
        records.append({"holding_month": month, "gross": gross, "traded": traded, "net": gross - COST * traded,
                        "invested": invested, "missing": int((~available).sum())})
    return pd.DataFrame(records).set_index("holding_month")


def spread_stats(s: pd.Series) -> dict:
    return {"annual_mean": 12 * s.mean(), "volatility": math.sqrt(12) * s.std(ddof=1),
            "sharpe": math.sqrt(12) * s.mean() / s.std(ddof=1), "max_drawdown": maximum_drawdown(s),
            "skewness": s.skew(), "excess_kurtosis": s.kurt(), "worst_month": s.min(),
            "mar_may_2009": float((1 + s.loc["2009-03":"2009-05"]).prod() - 1)}


DESIGNS = [("Base", 1, None, False), ("DM (momentum weight)", "k", None, False),
           ("DM fixed k (diagnostic)", "k_bar", None, False), ("BSC (portfolio volatility)", 1, None, True),
           ("HHZ (trend, cash)", 1, "cash", False), ("HHZ-R (trend, reweighted)", 1, "reweight", False),
           ("DM + HHZ", "k", "cash", False), ("BSC + HHZ", 1, "cash", True)]
# (label, momentum-weight multiplier: 1, k_t, or k_bar = its average over the window (diagnostic, ex post),
#  trend rule, portfolio volatility target: False, True = exposure capped at 1, or a number = exposure cap)
SUBPERIODS = [("2005-01", "2008-09"), ("2008-10", "2012-06"), ("2012-07", "2016-03"), ("2016-04", "2019-12")]


def bsc_replication(daily: Daily) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """WML (EW, VW), its BSC-scaled version, and the monthly volatility forecasts by holding month."""
    prices, universe = load_prices()
    mom = tradable_at_formation(momentum_signals(prices, universe)["MOM"])
    mom["decile"] = assign_deciles(mom, signal_column="signal").astype(int)
    caps, _ = load_formation_caps(CRSP_MONTHLY, mom[KEY], chunksize=500_000)
    mom = (mom.merge(caps, on=KEY, how="left").rename(columns={"formation_market_cap": "cap"})
           .merge(load_supplement([*KEY, "ret_final"]), on=KEY, how="left"))
    mom["holding_month"] = code_month(month_code(mom["formation_month"]) + 1)
    mom["mcode"] = month_code(mom["formation_month"]) + 1
    wml, sigmas, rows, weights = {}, {}, [], {}
    for weighting in ["EW", "VW"]:
        monthly = monthly_portfolios(mom, "ret_final", weighting == "VW")
        wml[weighting] = monthly[10] - monthly[1]
        legs = []
        for decile in [10, 1]:
            leg = mom.loc[mom["decile"].eq(decile) & (mom["cap"].gt(0) if weighting == "VW" else True)]
            members = leg[["PERMNO", "mcode"]].assign(w=leg["cap"] if weighting == "VW" else 1.0)
            members["w"] = members["w"] / members.groupby("mcode")["w"].transform("sum")
            legs.append(daily.portfolio(members))
        sigma = daily.forecast((legs[0] - legs[1]).dropna())
        sigma.index = code_month(sigma.index)
        sigmas[weighting] = sigma
        scale = (TARGET / sigma).reindex(wml[weighting].index)
        managed = scale * wml[weighting]
        weights[weighting] = scale
        for window, (first, last) in {"2005-2019": (START, END), "all available": ("2001-01", END)}.items():
            ok = scale.notna() & (scale.index >= first) & (scale.index <= last)
            for label, s in [("WML", wml[weighting][ok]), ("WML scaled (BSC)", managed[ok])]:
                rows.append({"weighting": weighting, "window": window, "portfolio": label, "months": int(ok.sum()),
                             "first": s.index.min(), **spread_stats(s),
                             "avg_weight": scale[ok].mean() if "scaled" in label else 1.0,
                             "min_weight": scale[ok].min() if "scaled" in label else 1.0,
                             "max_weight": scale[ok].max() if "scaled" in label else 1.0})
    monthly = pd.DataFrame({f"WML {w}": wml[w] for w in wml} | {f"weight {w}": weights[w] for w in weights})
    return pd.DataFrame(rows), monthly, sigmas


def select_top(frame: pd.DataFrame, portfolio: str) -> pd.DataFrame:
    """Decile 10 (``assign_deciles``: ties at the boundary go to the higher PERMNO) or the top 50 (score descending,
    PERMNO ascending, as ``top_n_returns``) of a frame with a ``score`` column. Shared by every product script."""
    if portfolio == "D10":
        return frame.loc[assign_deciles(frame, signal_column="score").astype(int) == 10]
    ordered = frame.sort_values(["formation_month", "score", "PERMNO"], ascending=[True, False, True])
    return ordered.groupby("formation_month", sort=True).head(50)


def run_product(frame: pd.DataFrame, base: dict[str, float], wml_sigma: pd.Series, daily: Daily, rf: pd.Series,
                last: str, ma: tuple[int, int] = (50, 200), designs=DESIGNS) -> dict[tuple[str, str], pd.DataFrame]:
    """Monthly results of every design for decile 10 and the top 50. ``base`` maps rank columns of ``frame`` to
    score weights and must include ``mom``; DM multiplies that weight by k_t = target / sigma_hat of the EW WML."""
    frame = frame.assign(holding_month=code_month(month_code(frame["formation_month"]) + 1),
                         mcode=month_code(frame["formation_month"]) + 1).reset_index(drop=True)
    k_month = TARGET / wml_sigma
    k = k_month.reindex(frame["holding_month"]).to_numpy()
    if np.isnan(k[frame["holding_month"].between(START, last).to_numpy()]).any():
        raise ValueError("Missing WML volatility forecast inside the evaluation window.")
    columns, weights, momentum = [frame[c] for c in base], list(base.values()), list(base).index("mom")

    def score(multiplier) -> np.ndarray:
        w = list(weights)
        w[momentum] = weights[momentum] * multiplier
        return combined_score(columns, w)

    scores = {1: score(1.0), "k": score(k), "k_bar": score(float(k_month.loc[START:last].mean()))}
    frame = frame.merge(daily.trend(*ma), on=KEY, how="left", validate="one_to_one")
    frame["no_uptrend"] = frame["no_uptrend"].fillna(True).astype(bool)  # no trend row: not a confirmed uptrend
    months = sorted(frame["holding_month"].unique())

    def weighted(chosen: pd.DataFrame, trend: str | None) -> pd.DataFrame:
        n = chosen.groupby("formation_month")["PERMNO"].transform("size")
        kept = ~chosen["no_uptrend"] if trend else pd.Series(True, index=chosen.index)
        if trend == "reweight":
            n = kept.groupby(chosen["formation_month"]).transform("sum").clip(lower=1)
        out = chosen.assign(w=np.where(kept, 1.0 / n, 0.0))
        return out.loc[out["w"].gt(0), ["formation_month", "holding_month", "mcode", "PERMNO", "w", "ret_final"]]

    def vol_scaled(holdings: pd.DataFrame, cap: float) -> pd.DataFrame:
        """Exposure min(cap, target / sigma_hat); cap = inf is BSC eq. 6 as written (exposure above 1 is borrowed at RF)."""
        sigma = daily.forecast(daily.portfolio(holdings[["PERMNO", "mcode", "w"]]))
        sigma.index = code_month(sigma.index)
        exposure = np.minimum(cap, TARGET / sigma).reindex(holdings["holding_month"]).fillna(1.0)
        return holdings.assign(w=holdings["w"] * exposure.to_numpy())

    results = {}
    for portfolio in ["D10", "Top 50"]:
        for label, multiplier, trend, scaled in designs:
            holdings = weighted(select_top(frame.assign(score=scores[multiplier]), portfolio), trend)
            cap = 1.0 if scaled is True else scaled  # True: long-only BSC (no leverage); a number: exposure cap
            monthly = managed_returns(vol_scaled(holdings, cap) if scaled else holdings, rf, months)
            results[(portfolio, label)] = monthly.loc[(monthly.index >= START) & (monthly.index <= last)]
    return results


def summary_row(portfolio: str, label: str, monthly: pd.DataFrame, rf: pd.Series, sp: pd.Series) -> dict:
    r = monthly["net"]
    return {"portfolio": portfolio, "design": label, "months": len(r), "avg_invested": monthly["invested"].mean(),
            "min_invested": monthly["invested"].min(), "turnover_one_way": monthly["traded"].mean() / 2,
            "missing_held": int(monthly["missing"].sum()), **metrics(r, rf.reindex(r.index), sp.reindex(r.index)),
            "year_2008": float((1 + r.loc["2008-01":"2008-12"]).prod() - 1),
            "mar_may_2009": float((1 + r.loc["2009-03":"2009-05"]).prod() - 1)}


def year_2020(r: pd.Series, invested: pd.Series | None = None) -> dict:
    def compound(first: str, last: str) -> float:
        return float((1 + r.loc[first:last]).prod() - 1)

    out = {"jan": compound("2020-01", "2020-01"), "feb_mar_crash": compound("2020-02", "2020-03"),
           "apr_dec_rebound": compound("2020-04", "2020-12"), "nov_vaccine": compound("2020-11", "2020-11"),
           "year_2020": compound("2020-01", "2020-12"), "max_drawdown_2020": maximum_drawdown(r.loc["2020-01":"2020-12"])}
    if invested is not None:
        out |= {f"invested_{m}": float(invested[f"2020-{m}"]) for m in ["03", "04", "05", "06"]}
    return out


def versus_base(results: dict, portfolio: str, label: str, rf: pd.Series) -> dict:
    """Paired block-bootstrap Sharpe difference against Base over 2005-2019 (exploratory; no multiple-test adjustment)."""
    if label == "Base":
        return {}
    diff = paired_sharpe_difference(results[(portfolio, label)]["net"].loc[START:END],
                                    results[(portfolio, "Base")]["net"].loc[START:END], rf)
    return {f"vs_base_{k}": v for k, v in diff.items()}


def subperiod_rows(portfolio: str, label: str, r: pd.Series, rf: pd.Series) -> list[dict]:
    rows = []
    for first, last in SUBPERIODS:
        part = r.loc[first:last]
        excess = part - rf.reindex(part.index)
        rows.append({"portfolio": portfolio, "design": label, "period": f"{first}..{last}", "months": len(part),
                     "annual_mean": 12 * part.mean(), "volatility": math.sqrt(12) * part.std(ddof=1),
                     "sharpe": math.sqrt(12) * excess.mean() / excess.std(ddof=1), "max_drawdown": maximum_drawdown(part)})
    return rows


def monthly_long(results: dict) -> pd.DataFrame:
    frames = {f"{p} | {l}": m[["gross", "net", "traded", "invested", "missing"]] for (p, l), m in results.items()}
    return pd.concat(frames, names=["series", "holding_month"]).reset_index()


def main(sp_path: str, covid: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(sp_path, dtype={"month": str}).set_index("month")["sp500tr_ret"]
    rf = load_french_monthly(FF5_MONTHLY_ZIP)["rf"]
    rf.index = rf.index.astype(str)
    daily = Daily()
    replication, wml_monthly, wml_sigma = bsc_replication(daily)
    pd.set_option("display.width", 250)
    s = sp.loc[START:END]
    sp_row = {"portfolio": "S&P 500 TR", "design": "index", **metrics(s, rf.reindex(s.index), s)}

    if covid:  # three-factor product (no CNN images after the 2019-12 formation), run on to December 2020
        panel, *_ = build_panel(FIRST_FORMATION, "2020-11")
        g = panel.groupby("formation_month")
        frame = panel.assign(mom_rank=g["mom"].transform(pct_rank), beta_rank=1 - g["beta"].transform(pct_rank),
                             ivol_rank=1 - g["ivol_daily"].transform(pct_rank))
        frame = frame.drop(columns="mom").rename(columns={"mom_rank": "mom"})
        results = run_product(frame, {"mom": 1 / 3, "beta_rank": 1 / 3, "ivol_rank": 1 / 3}, wml_sigma["EW"], daily, rf,
                              "2020-12")
        rows = [{**summary_row(p, l, m.loc[START:END], rf, sp), **versus_base(results, p, l, rf),
                 **year_2020(m["net"], m["invested"])} for (p, l), m in results.items()]
        table = pd.DataFrame(rows + [{**sp_row, **year_2020(sp)}])
        table.to_csv(OUTPUT_DIR / "covid_2020_three_factor.csv", index=False)
        monthly_long(results).to_csv(OUTPUT_DIR / "covid_2020_three_factor_monthly.csv", index=False)
        cols = ["portfolio", "design", "sharpe", "max_drawdown", "vs_base_sharpe_diff", "jan", "feb_mar_crash",
                "apr_dec_rebound", "year_2020", "invested_03", "invested_04"]
        print("three-factor product: sharpe, max_drawdown and vs_base are 2005-2019; the other columns are 2020")
        print(table[cols].round(3).to_string(index=False))
        return

    replication.to_csv(OUTPUT_DIR / "bsc_replication.csv", index=False)
    wml_monthly.to_csv(OUTPUT_DIR / "bsc_wml_monthly.csv")
    scores = pd.read_csv(CNN_SCORES, usecols=[*KEY, "cnn_prob_up"], dtype={"formation_month": str})
    panel, *_ = build_panel()
    four = panel.merge(scores, on=KEY, how="inner", validate="one_to_one")
    g = four.groupby("formation_month")
    four = four.assign(mom_rank=g["mom"].transform(pct_rank), beta_rank=1 - g["beta"].transform(pct_rank),
                       ivol_rank=1 - g["ivol_daily"].transform(pct_rank), cnn_rank=g["cnn_prob_up"].transform(pct_rank))
    four = four.drop(columns="mom").rename(columns={"mom_rank": "mom"})
    results = run_product(four, {"mom": 0.25, "beta_rank": 0.25, "ivol_rank": 0.25, "cnn_rank": 0.25}, wml_sigma["EW"],
                          daily, rf, END)
    rows = [{**summary_row(p, l, m, rf, sp), **versus_base(results, p, l, rf)} for (p, l), m in results.items()]
    product = pd.DataFrame(rows + [sp_row])
    product.to_csv(OUTPUT_DIR / "product_managed.csv", index=False)
    subperiods = [row for (p, l), m in results.items() for row in subperiod_rows(p, l, m["net"], rf)]
    subperiods += subperiod_rows("S&P 500 TR", "index", s, rf)
    pd.DataFrame(subperiods).to_csv(OUTPUT_DIR / "product_managed_subperiods.csv", index=False)
    monthly_long(results).to_csv(OUTPUT_DIR / "product_managed_monthly.csv", index=False)
    net = pd.DataFrame({f"{p} | {l}": m["net"] for (p, l), m in results.items()})
    net.assign(sp500tr=sp).loc[START:END].to_csv(OUTPUT_DIR / "product_managed_monthly_net.csv")
    show = ["portfolio", "design", "avg_invested", "min_invested", "turnover_one_way", "annual_mean", "volatility",
            "sharpe", "max_drawdown", "alpha_sp", "alpha_sp_t", "vs_base_sharpe_diff", "vs_base_ci95_low_6m",
            "vs_base_ci95_high_6m", "vs_base_ci95_low_12m", "vs_base_ci95_high_12m"]
    print(replication.round(3).to_string(index=False))
    print(product[show].round(3).to_string(index=False))
    table = pd.DataFrame(subperiods).pivot_table(index=["portfolio", "design"], columns="period", values="sharpe", sort=False)
    print(table.round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1], covid="--covid" in sys.argv[2:])
