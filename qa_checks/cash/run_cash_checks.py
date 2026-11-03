"""Reproduce Q2 fixed-exposure and lagged-volatility cash diagnostics.

Run from anywhere: python -B qa_checks/cash/run_cash_checks.py
Use --project-root to override the extracted project root and --out-dir to
choose a NEW output folder. Fixed exposure weights are ex-post diagnostics.
The lagged-volatility comparison uses gross returns on both sides.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--project-root', type=Path, default=HERE.parents[1])
parser.add_argument('--out-dir', type=Path, default=HERE / 'results_rerun')
args = parser.parse_args()
ROOT = args.project_root.resolve()
OUT = args.out_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'handoff' / 'ivol_combination' / 'src')]
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly
from performance_metrics import maximum_drawdown

source = ROOT / 'outputs/managed_portfolios/product_managed_monthly.csv'
raw = pd.read_csv(source, dtype={'holding_month': str})
data = raw.pivot(index='holding_month', columns='series', values=['net', 'gross', 'invested']).loc['2005-01':'2019-12']
base, overlay = data['net']['D10 | Base'], data['net']['D10 | BSC + HHZ']
invested = data['invested']['D10 | BSC + HHZ']
rf = load_french_monthly(FF5_MONTHLY_ZIP)['rf']
rf.index = rf.index.astype(str)
rf = rf.reindex(data.index)
assert len(data) == 180 and not pd.concat([base, overlay, invested, rf], axis=1).isna().any().any()

exposure = float(invested.mean())
target_vol = float(overlay.std(ddof=1))
def mix(e): return e * base + (1 - e) * rf
lo, hi = 0., 1.
assert mix(lo).std(ddof=1) <= target_vol <= mix(hi).std(ddof=1)
for _ in range(60):
    mid = (lo + hi) / 2
    if mix(mid).std(ddof=1) < target_vol: lo = mid
    else: hi = mid
risk_match = (lo + hi) / 2
series = {'Base': base, 'BSC+HHZ': overlay, 'Fixed same average exposure (ex post)': mix(exposure),
          'Fixed same realized volatility (ex post)': mix(risk_match)}
def metrics(r, cash):
    excess = r - cash
    return {'annual_mean': 12*r.mean(), 'annual_volatility': math.sqrt(12)*r.std(ddof=1),
            'sharpe': math.sqrt(12)*excess.mean()/excess.std(ddof=1), 'max_drawdown': maximum_drawdown(r)}
windows = [('full 2005-2019', data.index), ('2005-01..2008-09', data.index[:45]),
           ('2008-10..2012-06', data.index[45:90]), ('2012-07..2016-03', data.index[90:135]),
           ('2016-04..2019-12', data.index[135:])]
rows = [{'strategy': name, 'period': label, 'months': len(idx), **metrics(r.loc[idx], rf.loc[idx])}
        for name,r in series.items() for label,idx in windows]
pd.DataFrame(rows).to_csv(OUT / 'cash_comparison.csv', index=False)
pd.DataFrame({**series, 'RF': rf, 'BSC+HHZ invested': invested}).rename_axis('holding_month').to_csv(OUT / 'cash_monthly.csv')

base_gross, overlay_gross = data['gross']['D10 | Base'], data['gross']['D10 | BSC + HHZ']
base_lag = base_gross.rolling(12, min_periods=12).std(ddof=1).shift(1)
overlay_lag = overlay_gross.rolling(12, min_periods=12).std(ddof=1).shift(1)
weight = (overlay_lag/base_lag).clip(lower=0., upper=1.)
idx = data.index[12:]
assert weight.loc[idx].notna().all()
past = weight*base_gross + (1-weight)*rf
past_series = {'Base gross': base_gross, 'BSC+HHZ gross': overlay_gross,
               '12-month lagged volatility ratio gross': past}
past_rows = [{'strategy': name, 'period': '2006-01..2019-12', 'months': len(idx), **metrics(r.loc[idx], rf.loc[idx])}
             for name,r in past_series.items()]
pd.DataFrame(past_rows).to_csv(OUT / 'past_only_volatility_comparison.csv', index=False)
pd.DataFrame({**past_series, 'lagged_equity_weight': weight, 'RF': rf}).loc[idx].rename_axis('holding_month').to_csv(OUT / 'past_only_volatility_monthly.csv')
info = {'source': str(source), 'risk_free_source': str(FF5_MONTHLY_ZIP), 'months': 180,
        'same_average_exposure_ex_post': exposure, 'same_realized_volatility_ex_post': risk_match,
        'past_only_lookback_months': 12, 'past_only_average_equity_weight': float(weight.loc[idx].mean()),
        'limitations': ['fixed mixes use base net returns and omit separate stock/cash rebalancing cost',
                        'fixed exposures are ex post; lagged comparison is gross and risk is not exactly matched',
                        'no significance test or causal attribution']}
(OUT / 'method_and_inputs.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
print(pd.DataFrame(rows).query("period == 'full 2005-2019'").to_string(index=False))
print(pd.DataFrame(past_rows).to_string(index=False))
