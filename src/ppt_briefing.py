"""Official number table and figures (Korean and English) for the team briefing on BSC and HHZ (2026-09-30, v2).

Usage: python src/ppt_briefing.py
Reads saved portfolio-level aggregates only (outputs/managed_portfolios/, outputs/longonly_d10_vs_sp500/,
data/benchmarks/) and writes to handoff/Team_share_2026-09-30/:
  official_numbers.csv   one row per portfolio x design (2005-01..2019-12, EW, net of 0.1% per dollar traded)
  subperiods_final.csv   four 45-month subperiods for the official designs
  figures/*_ko.png, figures/*_en.png   figures; figures/data/*.csv holds the numbers behind each figure
Designs (2026-09-30): official = Base, BSC (formerly BSC-B, no leverage), HHZ (teammate trend rule), BSC + HHZ;
exploratory = BSC with leverage, HHZ-R; appendix only (not for the professor meeting) = DM (formerly BSC-A).
Colors: Okabe-Ito hues validated for color-vision deficiency; Base and the S&P 500 are neutral references.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_window import END, START  # noqa: E402

MP = ROOT / "outputs/managed_portfolios"
OUT = ROOT / "handoff/Team_share_2026-09-30"
FIG = OUT / "figures"
DATA = FIG / "data"
SOURCE = "outputs/managed_portfolios/product_managed.csv"
LEVERAGE = "outputs/managed_portfolios/leverage_bsc.csv"

# id suffix, source file, label in that file, English name, status, color
DESIGNS = [
    ("BASE", SOURCE, "Base", "Base", "official", "#1F2933"),
    ("BSC", SOURCE, "BSC (portfolio volatility)", "BSC", "official", "#0072B2"),
    ("HHZ", SOURCE, "HHZ (trend, cash)", "HHZ", "official", "#D55E00"),
    ("BSC_HHZ", SOURCE, "BSC + HHZ", "BSC + HHZ", "official", "#009E73"),
    ("BSC_LEV", LEVERAGE, "BSC no cap (paper)", "BSC with leverage (no cap)", "exploratory", "#56B4E9"),
    ("BSC_LEV15", LEVERAGE, "BSC cap 1.5", "BSC with leverage (cap 1.5)", "exploratory", "#56B4E9"),
    ("BSC_LEV_HHZ", LEVERAGE, "BSC no cap + HHZ", "BSC with leverage (no cap) + HHZ", "exploratory", "#56B4E9"),
    ("HHZ_R", SOURCE, "HHZ-R (trend, reweighted)", "HHZ-R (reweighted)", "exploratory", "#999999"),
    ("DM", SOURCE, "DM (momentum weight)", "DM (appendix)", "appendix", "#CC79A7"),
    ("DM_FIXED_K", SOURCE, "DM fixed k (diagnostic)", "DM fixed k (appendix, diagnostic)", "appendix", "#CC79A7"),
    ("DM_HHZ", SOURCE, "DM + HHZ", "DM + HHZ (appendix)", "appendix", "#E69F00"),
]
COLOR = {d[0]: d[5] for d in DESIGNS}
MAIN = ["BASE", "BSC", "HHZ", "BSC_HHZ"]
SP_COLOR, NEUTRAL = "#7A7A7A", "#1F2933"
PORTFOLIOS = ["D10", "Top 50"]
SELECTION = {"D10": "top decile of the combined score (assign_deciles), 1/N each",
             "Top 50": "top 50 by combined score (ties: lower PERMNO first), 1/50 each"}

TEXT = {
    "ko": {
        "port": {"D10": "D10 (상위 10%)", "Top 50": "상위 50종목"},
        "name": {"BASE": "Base (기본)", "BSC": "BSC", "HHZ": "HHZ", "BSC_HHZ": "BSC + HHZ", "BSC_LEV": "BSC 레버리지 (상한 없음)",
                 "SP": "S&P 500 TR"},
        "flow_title": "여섯 논문의 역할과 적용 위치",
        "flow": ["① 데이터: 교수님 CSV 종목(NYSE·AMEX·NASDAQ), 월별 수익률(상장폐지 보정)",
                 "② 네 가지 종목 선정 신호 (각 논문에서 가져온 것)",
                 "③ 결합 점수: 신호별 월간 순위(0~1)를 25%씩 합산",
                 "④ 종목 선정: 점수 상위 10%(D10) 또는 상위 50종목, 종목당 1/N",
                 "⑤ 투자비중·현금화: 주식에 넣지 않은 몫은 단기금리(FF RF)",
                 "⑥ 거래비용: 거래금액의 0.1% (매수·매도 모두)",
                 "⑦ 성과 평가: 샤프·MDD·회전율, S&P 500 총수익 대비 (2005-01~2019-12)"],
        "signals": ["모멘텀\nJegadeesh–Titman", "저베타\nFrazzini–Pedersen", "저IVOL\nAng 외", "CNN 차트\nJiang–Kelly–Xiu"],
        "side_team": "팀 자체 규칙 (탐색용, 논문 아님)\nBAND·TEAM_DR: 보유 종목 유지 구간",
        "side_hhz": "HHZ (Han–Huang–Zhou 롱 규칙)\n상승 추세 확인 안 된 종목 몫 → 현금",
        "side_bsc": "BSC (Barroso–Santa-Clara 식 5·6)\n포트폴리오 변동성 → 주식 비중 e_t ≤ 1",
        "growth_title": "{p}: 1달러 성장과 낙폭 (비용 0.1% 후, 2005-01~2019-12)",
        "growth_y": "1달러의 성장 (로그 눈금)", "dd_y": "고점 대비 낙폭 (%)",
        "growth_note": "월말 기준 월간 수익률로 계산. 낙폭은 각 시점까지의 최고치 대비 하락률이며, 가장 낮은 값이 MDD입니다.",
        "scatter_title": "샤프와 MDD로 본 설계 비교 (2005-01~2019-12, EW, 비용 0.1% 후)",
        "scatter_x": "최대낙폭 MDD (손실 크기, %) → 왼쪽일수록 좋음", "scatter_y": "샤프 (연율, 무위험 초과) → 위쪽일수록 좋음",
        "scatter_note": "속이 빈 원 = 레버리지 허용 BSC(탐색용). 나머지는 공식 후보.",
        "inv_y": "주식 투자비중 (%) · 나머지는 현금",
        "inv2008_title": "2008 금융위기 전후 주식 투자비중 (4신호, 레버리지 없음)",
        "inv2008_note": "HHZ는 상승 추세가 확인되지 않은 종목의 몫, BSC는 변동성 목표에 따른 축소분이 현금입니다.",
        "inv2020_title": "2020년 스트레스: 3신호 버전(CNN 없음)의 주식 투자비중",
        "inv2020_note": "평가 기간 밖이며 CNN이 없는 3신호 전략입니다. 4신호 전략의 연속 성과가 아닙니다.",
        "change_title": "Base 대비 변화: 샤프는 오른쪽, MDD는 오른쪽(0에 가까울수록)이 좋음",
        "change_panel": "{p} · {m} (세로선 = Base {v})", "sharpe": "샤프", "mdd": "MDD (%)",
        "bsc_title": "BSC 재현: 모멘텀 롱숏(EW)에 원 논문 규칙 적용 (2005-01~2019-12, 비용 전)",
        "bsc_raw": "모멘텀 승자-패자 (그대로)", "bsc_scaled": "BSC 스케일 (목표 변동성 연 12%)", "bsc_k": "노출 배수 k_t",
        "bsc_note": "k_t = 연 12% ÷ 직전 126거래일 일별 수익률로 추정한 변동성. 원 논문 재현용이며, 우리 롱온리 BSC는 같은 식을 "
                    "포트폴리오 자신의 변동성에 적용합니다.",
        "lev_title": "레버리지를 허용한 BSC의 주식 투자비중 (상한 없음, 4신호)",
        "lev_note": "100% 위는 무위험수익률로 빌려 투자한 부분입니다. 공식 대표 BSC는 100%에서 멈춥니다.",
        "lev_label": "레버리지 BSC (상한 없음)", "cap_label": "BSC (대표, 레버리지 없음)",
    },
    "en": {
        "port": {"D10": "D10 (top 10%)", "Top 50": "Top 50 stocks"},
        "name": {"BASE": "Base", "BSC": "BSC", "HHZ": "HHZ", "BSC_HHZ": "BSC + HHZ", "BSC_LEV": "BSC with leverage (no cap)",
                 "SP": "S&P 500 TR"},
        "flow_title": "The six papers and where each one enters",
        "flow": ["1 Data: professor's CSV stocks (NYSE, AMEX, NASDAQ), monthly returns with delisting",
                 "2 Four stock-selection signals (one from each paper)",
                 "3 Combined score: monthly rank (0-1) of each signal, 25% each",
                 "4 Selection: top 10% by score (D10) or top 50, 1/N each",
                 "5 Exposure and cash: money not in stocks earns the T-bill rate (FF RF)",
                 "6 Trading cost: 0.1% of the amount traded (buys and sells)",
                 "7 Evaluation: Sharpe, MDD, turnover, vs S&P 500 total return (2005-01 to 2019-12)"],
        "signals": ["Momentum\nJegadeesh-Titman", "Low beta\nFrazzini-Pedersen", "Low IVOL\nAng et al.", "CNN chart\nJiang-Kelly-Xiu"],
        "side_team": "Team rules (exploratory, not a paper)\nBAND, TEAM_DR: holding buffer",
        "side_hhz": "HHZ (Han-Huang-Zhou long rule)\nno confirmed uptrend -> weight to cash",
        "side_bsc": "BSC (Barroso-Santa-Clara eq. 5-6)\nportfolio volatility -> stock share e_t <= 1",
        "growth_title": "{p}: growth of $1 and drawdown (net of 0.1%, 2005-01 to 2019-12)",
        "growth_y": "Growth of $1 (log scale)", "dd_y": "Drawdown from peak (%)",
        "growth_note": "Monthly returns at month-end. Drawdown is the fall from the running peak; its lowest value is the MDD.",
        "scatter_title": "Designs by Sharpe and MDD (2005-01 to 2019-12, EW, net of 0.1%)",
        "scatter_x": "Maximum drawdown (loss, %) -> left is better", "scatter_y": "Sharpe (annualized, over RF) -> up is better",
        "scatter_note": "Hollow circle = BSC with leverage (exploratory). The rest are the official candidates.",
        "inv_y": "Share in stocks (%), rest in cash",
        "inv2008_title": "Share in stocks around the 2008 crisis (four signals, no leverage)",
        "inv2008_note": "Cash is the weight of stocks without a confirmed uptrend (HHZ) or the cut from the volatility target (BSC).",
        "inv2020_title": "2020 stress: share in stocks, three-signal version (no CNN)",
        "inv2020_note": "Outside the evaluation window, three signals without CNN. Not a continuation of the four-signal strategy.",
        "change_title": "Change vs Base: right is better for Sharpe and for MDD (closer to zero)",
        "change_panel": "{p} - {m} (line = Base {v})", "sharpe": "Sharpe", "mdd": "MDD (%)",
        "bsc_title": "BSC replication: the paper's rule on momentum long-short (EW, 2005-01 to 2019-12, before costs)",
        "bsc_raw": "Momentum winners minus losers", "bsc_scaled": "BSC scaled (target 12% a year)", "bsc_k": "Exposure k_t",
        "bsc_note": "k_t = 12% / volatility forecast from the last 126 daily returns. This is the replication; our long-only BSC "
                    "applies the same formula to the portfolio's own volatility.",
        "lev_title": "Share in stocks of BSC with leverage allowed (no cap, four signals)",
        "lev_note": "Above 100% is borrowed at the risk-free rate. The representative BSC stops at 100%.",
        "lev_label": "BSC with leverage (no cap)", "cap_label": "BSC (representative, no leverage)",
    },
}

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "axes.edgecolor": "#6B7280",
                     "axes.labelcolor": "#1F2933", "xtick.color": "#374151", "ytick.color": "#374151",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def drawdown_dates(r: pd.Series) -> tuple[str, str, str]:
    """Peak (last month-end before the fall), trough and recovery month of the maximum drawdown."""
    path = np.concatenate([[1.0], np.cumprod(1.0 + r.to_numpy())])
    labels = ["before start"] + list(r.index)
    trough = int(np.argmin(path / np.maximum.accumulate(path) - 1.0))
    peak = int(np.argmax(path[: trough + 1]))
    later = np.flatnonzero(path[trough:] >= path[peak])
    recovery = labels[trough + later[0]] if len(later) else "not recovered by 2019-12"
    return labels[peak], labels[trough], recovery


def load() -> tuple[dict, dict, pd.Series]:
    """Summary rows and monthly results by source file; the leverage file keeps only the four-signal sample."""
    summary = {SOURCE: pd.read_csv(ROOT / SOURCE), LEVERAGE: pd.read_csv(ROOT / LEVERAGE)}
    main = pd.read_csv(MP / "product_managed_monthly.csv", dtype={"holding_month": str})
    lev = pd.read_csv(MP / "leverage_bsc_monthly.csv", dtype={"holding_month": str})
    monthly = {SOURCE: main, LEVERAGE: lev[lev["sample"] == "Equal 25%"]}
    sp = pd.read_csv(ROOT / "data/benchmarks/sp500_monthly.csv", dtype={"month": str}).set_index("month")["sp500tr_ret"]
    return summary, monthly, sp.loc[START:END]


def series(monthly: dict, source: str, portfolio: str, label: str, column: str = "net") -> pd.Series:
    m = monthly[source]
    return m[m["series"] == f"{portfolio} | {label}"].set_index("holding_month")[column].loc[START:END]


def official_table(summary: dict, monthly: dict, sp: pd.Series) -> pd.DataFrame:
    rows = []
    for portfolio in PORTFOLIOS:
        base = summary[SOURCE].query("portfolio == @portfolio and design == 'Base'").iloc[0]
        for short, source, label, name, status, _ in DESIGNS:
            r = summary[source][(summary[source].portfolio == portfolio) & (summary[source].design == label)].iloc[0]
            peak, trough, recovery = drawdown_dates(series(monthly, source, portfolio, label))
            rows.append({"id": f"{portfolio.replace(' ', '')}_{short}", "status": status, "portfolio": portfolio, "design": name,
                         "selection": SELECTION[portfolio], "weighting": "EW", "period": f"{START}..{END}",
                         "cost": "0.1% per dollar traded" + ("; borrowing at RF" if source == LEVERAGE else ""), "source": source,
                         "annual_mean": r.annual_mean, "cagr": r.cagr, "volatility": r.volatility, "sharpe": r.sharpe,
                         "max_drawdown": r.max_drawdown, "dd_peak": peak, "dd_trough": trough, "dd_recovery": recovery,
                         "turnover_one_way": r.turnover_one_way, "avg_invested": r.avg_invested, "min_invested": r.min_invested,
                         "max_invested": r.get("max_invested"),
                         "year_2008": r.year_2008, "mar_may_2009": r.mar_may_2009, "alpha_sp": r.alpha_sp, "alpha_sp_t": r.alpha_sp_t,
                         "sharpe_spread1": r.get("sharpe_spread1"), "max_drawdown_spread1": r.get("max_drawdown_spread1"),
                         "d_sharpe_vs_base": r.sharpe - base.sharpe, "d_mdd_vs_base_pp": 100 * (r.max_drawdown - base.max_drawdown),
                         "d_turnover_vs_base_pp": 100 * (r.turnover_one_way - base.turnover_one_way),
                         "boot_sharpe_diff": r.get("vs_base_sharpe_diff"), "boot_ci95_low_6m": r.get("vs_base_ci95_low_6m"),
                         "boot_ci95_high_6m": r.get("vs_base_ci95_high_6m"), "boot_ci95_low_12m": r.get("vs_base_ci95_low_12m"),
                         "boot_ci95_high_12m": r.get("vs_base_ci95_high_12m")})
    s = summary[SOURCE].query("portfolio == 'S&P 500 TR'").iloc[0]
    peak, trough, recovery = drawdown_dates(sp)
    rows.append({"id": "SP500TR", "status": "benchmark", "portfolio": "S&P 500 TR", "design": "S&P 500 total return",
                 "selection": "Yahoo ^SP500TR month-end", "weighting": "index", "period": f"{START}..{END}", "cost": "none",
                 "source": "data/benchmarks/sp500_monthly.csv", "annual_mean": s.annual_mean, "cagr": s.cagr,
                 "volatility": s.volatility, "sharpe": s.sharpe, "max_drawdown": s.max_drawdown, "dd_peak": peak,
                 "dd_trough": trough, "dd_recovery": recovery})
    lo = pd.read_csv(ROOT / "outputs/longonly_d10_vs_sp500/longonly_d10_vs_sp500.csv")
    for portfolio, weighting in [("D10", "EW"), ("Top 50", "top50 EW")]:
        for name, label, short in [("CNN only (matched)", "Vanilla CNN (CNN only)", "VANILLA_CNN"),
                                   ("3 signals", "3 signals (no CNN)", "THREE_SIGNALS")]:
            r = lo[(lo.portfolio == name) & (lo.weighting == weighting) & (lo.returns == "net 0.10%")].iloc[0]
            rows.append({"id": f"{portfolio.replace(' ', '')}_{short}", "status": "context", "portfolio": portfolio,
                         "design": label, "selection": SELECTION[portfolio], "weighting": "EW", "period": f"{START}..{END}",
                         "cost": "0.1% per dollar traded", "source": "outputs/longonly_d10_vs_sp500/longonly_d10_vs_sp500.csv",
                         "annual_mean": r.annual_mean, "cagr": r.cagr, "volatility": r.volatility, "sharpe": r.sharpe,
                         "max_drawdown": r.max_drawdown, "turnover_one_way": r.turnover_one_way, "alpha_sp": r.alpha_sp,
                         "alpha_sp_t": r.alpha_sp_t, "avg_invested": 1.0})
    return pd.DataFrame(rows)


def save(fig, name: str, lang: str, data: pd.DataFrame) -> None:
    fig.savefig(FIG / f"{name}_{lang}.png", dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    data.to_csv(DATA / f"{name}.csv")


def flow_figure(lang: str) -> None:
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(11.5, 8.4))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    def box(x, y, w, h, text, face="#F5F7F9", edge="#6B7280", style="-", size=9.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=face,
                                    edgecolor=edge, linewidth=1.3, linestyle=style))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color="#1F2933", linespacing=1.35)

    def arrow(x0, y0, x1, y1, color="#6B7280", style="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, color=color,
                                     linewidth=1.4, linestyle=style))

    x, w, h = 0.2, 6.6, 0.6
    ax.text(0.2, 8.42, t["flow_title"], fontsize=14, weight="bold", va="top", color="#1F2933")
    box(x, 7.15, w, h, t["flow"][0], size=9)
    box(x, 5.5, w, 1.35, "", face="#FFFFFF")
    ax.text(x + 0.15, 6.62, t["flow"][1], fontsize=9.4, va="center", color="#1F2933")
    for text, sx in zip(t["signals"], [0.35, 1.95, 3.55, 5.15]):
        box(sx, 5.62, 1.5, 0.78, text, face="#E8F0F7", edge="#1E5A86", size=8.8)
    steps = list(zip([4.55, 3.6, 2.65, 1.7, 0.75], t["flow"][2:]))
    for y, text in steps:
        box(x, y, w, h, text)
    for y0, y1 in [(7.15, 6.87), (5.5, 5.17), (4.55, 4.22), (3.6, 3.27), (2.65, 2.32), (1.7, 1.37)]:
        arrow(x + w / 2, y0, x + w / 2, y1)
    sx, sw, right = 7.25, 4.05, x + w + 0.03
    box(sx, 3.5, sw, 0.8, t["side_team"], face="#FFFFFF", edge="#9CA3AF", style="--", size=8.6)
    arrow(sx, 3.9, right, 3.9, color="#9CA3AF", style="--")
    box(sx, 2.55, sw, 0.8, t["side_hhz"], face="#FBEDE6", edge="#D55E00", size=8.9)
    arrow(sx, 2.95, right, 2.98, color="#D55E00")
    box(sx, 1.6, sw, 0.8, t["side_bsc"], face="#E6F1F8", edge="#0072B2", size=8.9)
    arrow(sx, 2.0, right, 2.8, color="#0072B2")
    save(fig, "fig1_six_papers_flow", lang, pd.DataFrame({"step": [s for _, s in steps]}))


def growth_figure(monthly: dict, sp: pd.Series, lang: str) -> None:
    t = TEXT[lang]
    for portfolio in PORTFOLIOS:
        lines = {short: series(monthly, src, portfolio, label) for short, src, label, *_ in DESIGNS if short in MAIN}
        lines["SP"] = sp
        index = lines["BASE"].index
        dates = pd.PeriodIndex(index, freq="M").to_timestamp(how="end")
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        data = {}
        for short, r in lines.items():
            wealth = (1 + r.reindex(index)).cumprod()
            dd = wealth / np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:] - 1
            style, color = ("--", SP_COLOR) if short == "SP" else ("-", COLOR[short])
            width = 2.6 if short == "BASE" else 2.0
            name = t["name"][short]
            top.plot(dates, wealth, color=color, linestyle=style, linewidth=width, label=f"{name}: ${wealth.iloc[-1]:.2f}")
            bottom.plot(dates, 100 * dd, color=color, linestyle=style, linewidth=width - 0.4)
            data[f"{short} wealth"], data[f"{short} drawdown"] = wealth.to_numpy(), dd.to_numpy()
        top.set_yscale("log")
        top.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
        top.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        top.set_ylabel(t["growth_y"])
        top.set_title(t["growth_title"].format(p=t["port"][portfolio]), fontsize=12, loc="left")
        top.legend(loc="upper left", frameon=False, fontsize=9)
        top.grid(axis="y", color="#E5E7EB")
        bottom.set_ylabel(t["dd_y"])
        bottom.grid(axis="y", color="#E5E7EB")
        bottom.axhline(0, color="#9CA3AF", linewidth=0.8)
        fig.text(0.01, -0.01, t["growth_note"], fontsize=8.5, color="#4B5563")
        fig.tight_layout()
        save(fig, f"fig2_growth_drawdown_{portfolio.replace(' ', '')}", lang, pd.DataFrame(data, index=index))


def scatter_figure(table: pd.DataFrame, lang: str) -> None:
    t = TEXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    sp = table[table.id == "SP500TR"].iloc[0]
    rows = []
    for ax, portfolio in zip(axes, PORTFOLIOS):
        for short in MAIN + ["BSC_LEV"]:
            r = table[table.id == f"{portfolio.replace(' ', '')}_{short}"].iloc[0]
            x, y = -100 * r.max_drawdown, r.sharpe
            hollow = short == "BSC_LEV"
            ax.scatter(x, y, s=90, marker="s" if short == "BASE" else "o", facecolor="white" if hollow else COLOR[short],
                       edgecolor=COLOR[short] if hollow else "white", linewidth=2.0 if hollow else 1.5, zorder=3)
            # neighbours sit close together: label BSC + HHZ and BSC on their left
            offset, ha = {"BSC_HHZ": ((-6, -16), "right"), "BSC": ((-6, 5), "right")}.get(short, ((6, 5), "left"))
            ax.annotate(t["name"][short], (x, y), xytext=offset, textcoords="offset points", ha=ha, fontsize=8.8, color="#1F2933")
            rows.append({"portfolio": portfolio, "design": short, "mdd_loss_pct": x, "sharpe": y})
        ax.scatter(-100 * sp.max_drawdown, sp.sharpe, s=80, marker="D", color=SP_COLOR, zorder=3)
        ax.annotate("S&P 500 TR", (-100 * sp.max_drawdown, sp.sharpe), xytext=(-70, 6), textcoords="offset points", fontsize=8.8)
        ax.set_title(t["port"][portfolio], fontsize=11.5, loc="left")
        ax.set_xlabel(t["scatter_x"])
        ax.grid(color="#E5E7EB")
        ax.set_xlim(12, 55)
    axes[0].set_ylabel(t["scatter_y"])
    fig.suptitle(t["scatter_title"], fontsize=12.5, x=0.01, ha="left")
    fig.text(0.01, -0.02, t["scatter_note"], fontsize=8.5, color="#4B5563")
    fig.tight_layout()
    save(fig, "fig3_sharpe_mdd", lang, pd.DataFrame(rows))


def invested_figure(frame: pd.DataFrame, first: str, last: str, title: str, note: str, name: str, lang: str) -> None:
    t = TEXT[lang]
    wide = frame.pivot(index="holding_month", columns="series", values="invested").loc[first:last]
    dates = pd.PeriodIndex(wide.index, freq="M").to_timestamp(how="start")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    data = {}
    for ax, portfolio in zip(axes, PORTFOLIOS):
        for short, _, label, *_ in DESIGNS:
            if short not in ("BSC", "HHZ", "BSC_HHZ"):
                continue
            s = 100 * wide[f"{portfolio} | {label}"]
            ax.step(dates, s, where="post", color=COLOR[short], linewidth=2.0, label=t["name"][short])
            data[f"{portfolio} {short}"] = s.to_numpy()
        ax.set_title(t["port"][portfolio], fontsize=11, loc="left")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#E5E7EB")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
    axes[0].set_ylabel(t["inv_y"])
    axes[1].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=12.5, x=0.01, ha="left")
    fig.text(0.01, -0.03, note, fontsize=8.5, color="#4B5563")
    fig.tight_layout()
    save(fig, name, lang, pd.DataFrame(data, index=wide.index))


def leverage_figure(monthly: dict, lang: str) -> None:
    t = TEXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    data = {}
    for ax, portfolio in zip(axes, PORTFOLIOS):
        lev = series(monthly, LEVERAGE, portfolio, "BSC no cap (paper)", "invested")
        cap = series(monthly, SOURCE, portfolio, "BSC (portfolio volatility)", "invested")
        dates = pd.PeriodIndex(lev.index, freq="M").to_timestamp(how="start")
        ax.step(dates, 100 * lev, where="post", color=COLOR["BSC_LEV"], linewidth=1.8, label=t["lev_label"])
        ax.step(dates, 100 * cap, where="post", color=COLOR["BSC"], linewidth=1.8, label=t["cap_label"])
        ax.axhline(100, color="#9CA3AF", linewidth=0.9, linestyle=":")
        ax.set_title(t["port"][portfolio], fontsize=11, loc="left")
        ax.grid(axis="y", color="#E5E7EB")
        data[f"{portfolio} no cap"], data[f"{portfolio} cap 1"] = lev.to_numpy(), cap.to_numpy()
    axes[0].set_ylabel(t["inv_y"])
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(t["lev_title"], fontsize=12.5, x=0.01, ha="left")
    fig.text(0.01, -0.03, t["lev_note"], fontsize=8.5, color="#4B5563")
    fig.tight_layout()
    save(fig, "fig7_leverage_exposure", lang, pd.DataFrame(data, index=lev.index))


def change_figure(table: pd.DataFrame, lang: str) -> None:
    t = TEXT[lang]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6))
    rows = []
    shorts = ["BSC", "HHZ", "BSC_HHZ", "BSC_LEV"]
    for i, portfolio in enumerate(PORTFOLIOS):
        part = table.set_index("id")
        key = portfolio.replace(" ", "")
        base = part.loc[f"{key}_BASE"]
        names = [t["name"][s] for s in shorts]
        for j, (metric, label, scale) in enumerate([("sharpe", t["sharpe"], 1.0), ("max_drawdown", t["mdd"], 100.0)]):
            ax = axes[i, j]
            values = [scale * part.loc[f"{key}_{s}", metric] for s in shorts]
            bars = ax.barh(names, [v - scale * base[metric] for v in values], left=scale * base[metric],
                           color=[COLOR[s] for s in shorts], height=0.55)
            bars[-1].set_hatch("//")
            bars[-1].set_edgecolor("white")
            ax.axvline(scale * base[metric], color=NEUTRAL, linewidth=1.6)
            for k, v in enumerate(values):
                ax.text(v, k, f" {v:.3f}" if metric == "sharpe" else f" {v:.1f}%", va="center",
                        ha="left" if v >= scale * base[metric] else "right", fontsize=8.8)
                rows.append({"portfolio": portfolio, "design": shorts[k], "metric": metric, "value": values[k] / scale,
                             "base": base[metric]})
            ax.invert_yaxis()
            shown = f"{scale * base[metric]:.3f}" if metric == "sharpe" else f"{scale * base[metric]:.1f}%"
            ax.set_title(t["change_panel"].format(p=t["port"][portfolio], m=label, v=shown), fontsize=10, loc="left")
            ax.grid(axis="x", color="#E5E7EB")
            ax.set_xlim((-40, -10) if metric == "max_drawdown" else (0.9, 1.3))
    fig.suptitle(t["change_title"], fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig5_change_vs_base", lang, pd.DataFrame(rows))


def bsc_figure(lang: str) -> None:
    t = TEXT[lang]
    w = pd.read_csv(MP / "bsc_wml_monthly.csv", index_col=0)
    w.index = w.index.astype(str)
    w = w.loc[START:END]
    dates = pd.PeriodIndex(w.index, freq="M").to_timestamp(how="end")
    raw, k = w["WML EW"], w["weight EW"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    top.plot(dates, (1 + raw).cumprod(), color=NEUTRAL, linewidth=2.2, label=t["bsc_raw"])
    top.plot(dates, (1 + k * raw).cumprod(), color="#0072B2", linewidth=2.2, label=t["bsc_scaled"])
    top.set_yscale("log")
    top.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    top.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    top.set_title(t["bsc_title"], fontsize=12, loc="left")
    top.legend(loc="upper left", frameon=False, fontsize=9)
    top.grid(axis="y", color="#E5E7EB")
    bottom.plot(dates, k, color="#0072B2", linewidth=1.8)
    bottom.axhline(1.0, color="#9CA3AF", linewidth=0.9, linestyle=":")
    bottom.set_ylabel(t["bsc_k"])
    bottom.grid(axis="y", color="#E5E7EB")
    fig.text(0.01, -0.01, t["bsc_note"], fontsize=8.5, color="#4B5563")
    fig.tight_layout()
    save(fig, "fig6_bsc_replication", lang, pd.DataFrame({"wml_ew": raw, "k_t": k, "wml_scaled": k * raw}))


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    summary, monthly, sp = load()
    table = official_table(summary, monthly, sp)
    table.to_csv(OUT / "official_numbers.csv", index=False)
    sub = pd.read_csv(MP / "product_managed_subperiods.csv")
    keep = {label: name for short, src, label, name, *_ in DESIGNS if short in MAIN} | {"index": "S&P 500 total return"}
    sub[sub.design.isin(keep)].assign(design=lambda d: d.design.map(keep)).to_csv(OUT / "subperiods_final.csv", index=False)
    covid = pd.read_csv(MP / "covid_2020_three_factor_monthly.csv", dtype={"holding_month": str})
    for lang in TEXT:
        t = TEXT[lang]
        flow_figure(lang)
        growth_figure(monthly, sp, lang)
        scatter_figure(table, lang)
        invested_figure(monthly[SOURCE], "2007-01", "2010-12", t["inv2008_title"], t["inv2008_note"], "fig4_invested_2008", lang)
        invested_figure(covid, "2019-07", "2020-12", t["inv2020_title"], t["inv2020_note"], "fig4b_invested_2020_three_factor", lang)
        change_figure(table, lang)
        bsc_figure(lang)
        leverage_figure(monthly, lang)
    pd.set_option("display.width", 250)
    cols = ["id", "status", "annual_mean", "volatility", "sharpe", "max_drawdown", "dd_peak", "dd_trough", "dd_recovery",
            "turnover_one_way", "avg_invested", "d_sharpe_vs_base", "d_mdd_vs_base_pp"]
    print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
