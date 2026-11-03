"""Ex-post diagnostic comparisons for D10 BSC+HHZ using saved monthly returns.

Fixed cash benchmarks approximate a monthly 82/18 or risk-matched mix from
the base net return and the monthly risk-free return. Extra trading costs of
rebalancing between cash and stocks are not represented. Neither mix is a
point-in-time portfolio rule: both exposure levels use full-sample outcomes.
"""
from pathlib import Path
import json
import math
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'handoff' / 'ivol_combination' / 'src')]
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly
from performance_metrics import maximum_drawdown

HERE = Path(__file__).resolve().parent
source = ROOT / 'outputs/managed_portfolios/product_managed_monthly.csv'
raw = pd.read_csv(source, dtype={'holding_month': str})
data = raw.pivot(index='holding_month', columns='series', values=['net', 'gross', 'invested'])
data = data.loc['2005-01':'2019-12']
base = data['net']['D10 | Base']
overlay = data['net']['D10 | BSC + HHZ']
invested = data['invested']['D10 | BSC + HHZ']
rf = load_french_monthly(FF5_MONTHLY_ZIP)['rf']
rf.index = rf.index.astype(str)
rf = rf.reindex(data.index)
assert len(data) == 180 and not pd.concat([base, overlay, invested, rf], axis=1).isna().any().any()

exposure = float(invested.mean())
target_vol = float(overlay.std(ddof=1))
def mix(e):
    return e * base + (1 - e) * rf
lo, hi = 0.0, 1.0
assert mix(lo).std(ddof=1) <= target_vol <= mix(hi).std(ddof=1)
for _ in range(60):
    mid = (lo + hi) / 2
    if mix(mid).std(ddof=1) < target_vol:
        lo = mid
    else:
        hi = mid
risk_match = (lo + hi) / 2
series = {'Base': base, 'BSC+HHZ': overlay,
          'Fixed same average exposure (ex post)': mix(exposure),
          'Fixed same realized volatility (ex post)': mix(risk_match)}

def metrics(r, cash):
    excess = r - cash
    return {'annual_mean': 12*r.mean(), 'annual_volatility': math.sqrt(12)*r.std(ddof=1),
            'sharpe': math.sqrt(12)*excess.mean()/excess.std(ddof=1),
            'max_drawdown': maximum_drawdown(r)}

rows = []
for name, r in series.items():
    for label, idx in [('full 2005-2019', data.index), ('2005-01..2008-09', data.index[:45]),
                       ('2008-10..2012-06', data.index[45:90]), ('2012-07..2016-03', data.index[90:135]),
                       ('2016-04..2019-12', data.index[135:])]:
        rows.append({'strategy': name, 'period': label, 'months': len(idx), **metrics(r.loc[idx], rf.loc[idx])})
pd.DataFrame(rows).to_csv(HERE / 'cash_comparison.csv', index=False)
pd.DataFrame({**series, 'RF': rf, 'BSC+HHZ invested': invested}).rename_axis('holding_month').to_csv(HERE / 'cash_monthly.csv')

# Exploratory past-information-only comparison. Monthly exposure is the ratio
# of the two portfolios' trailing 12-month gross volatilities, lagged by one
# month and capped at 100%. Both sides are gross because exact constituent-level
# trading costs of the counterfactual mix cannot be recovered from aggregates.
base_gross = data['gross']['D10 | Base']
overlay_gross = data['gross']['D10 | BSC + HHZ']
base_lag_vol = base_gross.rolling(12, min_periods=12).std(ddof=1).shift(1)
overlay_lag_vol = overlay_gross.rolling(12, min_periods=12).std(ddof=1).shift(1)
past_weight = (overlay_lag_vol / base_lag_vol).clip(lower=0.0, upper=1.0)
test_index = data.index[12:]
assert past_weight.loc[test_index].notna().all()
past_mix = past_weight*base_gross + (1-past_weight)*rf
past_series = {'Base gross': base_gross, 'BSC+HHZ gross': overlay_gross,
               '12-month lagged volatility ratio gross': past_mix}
past_rows = [{'strategy': name, 'period': '2006-01..2019-12', 'months': len(test_index),
              **metrics(r.loc[test_index], rf.loc[test_index])} for name,r in past_series.items()]
pd.DataFrame(past_rows).to_csv(HERE / 'past_only_volatility_comparison.csv', index=False)
pd.DataFrame({**past_series, 'lagged_equity_weight': past_weight, 'RF': rf}).loc[test_index].rename_axis('holding_month').to_csv(HERE / 'past_only_volatility_monthly.csv')
info = {'source': str(source), 'risk_free_source': str(FF5_MONTHLY_ZIP),
        'holding_period': ['2005-01', '2019-12'], 'months': len(data),
        'same_average_exposure_ex_post': exposure, 'same_realized_volatility_ex_post': risk_match,
        'risk_target_is_full_sample_ex_post': True,
        'past_only_diagnostic': {'period': ['2006-01','2019-12'], 'lookback_months': 12,
              'weight': 'min(1, lagged 12-month overlay gross volatility / lagged 12-month base gross volatility)',
              'average_equity_weight': float(past_weight.loc[test_index].mean()),
              'basis': 'gross; no trading costs on either comparator',
              'caution': 'lagged risk scaling is not guaranteed to equal realized volatility and is exploratory'},
        'limitations': ['fixed mixes use base net returns and omit separate stock/cash rebalancing cost',
                        'both exposure levels use the full sample and are descriptive only',
                        'no hypothesis test; not an independently risk-matched real-time strategy']}
(HERE / 'method_and_inputs.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(info, ensure_ascii=False, indent=2))
print(pd.DataFrame(rows).query("period == 'full 2005-2019'").to_string(index=False))
print(pd.DataFrame(past_rows).to_string(index=False))
