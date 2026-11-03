"""Reaggregate saved CNN-only outcomes into final report's four holding windows."""
from pathlib import Path
import sys
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(1, str(ROOT / 'handoff/ivol_combination/src'))
from ivol_specifications import FF5_MONTHLY_ZIP, load_french_monthly
from performance_metrics import maximum_drawdown, newey_west_t_stat
from delisting_robustness import cnn_universe

PERIODS = [('2005-01','2008-09'),('2008-10','2012-06'),('2012-07','2016-03'),('2016-04','2019-12')]
NET = ROOT / 'submission/IFE_Group7_Code/outputs/longonly_d10_vs_sp500/longonly_d10_monthly_net.csv'

def metrics(r, rf):
    excess = r - rf.reindex(r.index)
    if len(r) != 45 or excess.isna().any():
        raise ValueError('Expected 45 nonmissing monthly returns and RF values')
    return dict(months=len(r), annual_mean=12*r.mean(), annual_vol=np.sqrt(12)*r.std(ddof=1),
                sharpe=np.sqrt(12)*excess.mean()/excess.std(ddof=1), mdd=maximum_drawdown(r))

def hac_ols_period_difference(y, lags=4):
    # Four period intercepts (no shared intercept); sandwich with Bartlett HAC lag 4.
    y = y.reindex(pd.period_range('2005-01','2019-12',freq='M').astype(str))
    if len(y)!=180 or y.isna().any():
        raise ValueError('Spread comparison needs all 180 months')
    x = np.repeat(np.eye(4),45,axis=0)
    b = np.linalg.solve(x.T@x,x.T@y.to_numpy())
    score = x*(y.to_numpy()-x@b)[:,None]
    meat = score.T@score
    for k in range(1,lags+1):
        cross = score[k:].T@score[:-k]
        meat += (1-k/(lags+1))*(cross+cross.T)
    bread = np.linalg.inv(x.T@x)
    cov = bread@meat@bread
    c = np.array([-1,0,0,1])
    se = np.sqrt(c@cov@c)
    return {'annual_difference_last_minus_first':12*float(c@b), 'nw4_t':float((c@b)/se), 'monthly_se':float(se)}

net = pd.read_csv(NET,dtype={'holding_month':str}).set_index('holding_month')
rf = load_french_monthly(FF5_MONTHLY_ZIP)['rf']
rf.index=rf.index.astype(str)
series=['CNN only (matched) | EW','CNN only (matched) | top50 EW','CNN 70% | top50 EW','sp500tr']
rows=[]
for start,end in PERIODS:
    for name in series:
        r=net.loc[start:end,name].astype(float)
        rows.append({'period':f'{start}..{end}','series':name,**metrics(r,rf)})
pd.DataFrame(rows).to_csv(OUT/'portfolio_period_metrics.csv',index=False)

# Run the project's existing formation tradability, decile ranking and corrected-return merge.
f=cnn_universe()
f=f.loc[f.holding_month.between('2005-01','2019-12')].copy()
f['has_ret']=f.ret_final.notna()
coverage=f.groupby('holding_month').agg(scored=('PERMNO','size'),usable=('has_ret','sum'))
g=f.loc[f.has_ret].copy()
g['correct']=((g.cnn_prob_up>=.5)==(g.ret_final>0)).astype(float)
g['positive']=(g.ret_final>0).astype(float)
monthly_accuracy=g.groupby('holding_month').agg(accuracy=('correct','mean'),positive_share=('positive','mean'))
wide=g.groupby(['holding_month','cnn_decile'])['ret_final'].mean().unstack('cnn_decile')
if not {1,10}.issubset(wide.columns):
    raise ValueError('Missing CNN decile 1 or 10')
monthly=coverage.join(monthly_accuracy).join(wide[[1,10]].rename(columns={1:'q1',10:'q10'}))
monthly['q10_minus_q1']=monthly.q10-monthly.q1
if len(monthly)!=180 or monthly[['q1','q10','accuracy']].isna().any().any():
    raise ValueError('Missing a month or decile in signal diagnostics')
monthly.to_csv(OUT/'cnn_universe_monthly_diagnostics.csv',index_label='holding_month')
diagnostics=[]
for start,end in PERIODS:
    t=monthly.loc[start:end]
    diagnostics.append({'period':f'{start}..{end}','months':len(t), 'avg_scored':t.scored.mean(),
       'avg_usable':t.usable.mean(),'excluded_missing_total':int((t.scored-t.usable).sum()),
       'accuracy_equal_month':t.accuracy.mean(),'positive_share_equal_month':t.positive_share.mean(),
       'q1_annual_mean':12*t.q1.mean(),'q10_annual_mean':12*t.q10.mean(),
       'q10_minus_q1_annual_mean':12*t.q10_minus_q1.mean(),
       'q10_minus_q1_nw4_t':newey_west_t_stat(t.q10_minus_q1,lag=4)})
pd.DataFrame(diagnostics).to_csv(OUT/'cnn_universe_period_diagnostics.csv',index=False)
comparison=hac_ols_period_difference(monthly.q10_minus_q1)
(OUT/'period_test_metadata.json').write_text(json.dumps({'periods':PERIODS,'portfolio_input':str(NET),
 'ff5_rf_input':str(FF5_MONTHLY_ZIP),'signal_input':str(ROOT/'outputs/cnn_oos_deciles_2001_2019/cnn_assignments_with_outcomes.csv.gz'),
 'returns_input':str(ROOT/'handoff/delisting_supplement/next_month_returns_with_delisting_2000-2020.csv.gz'),
 'formation_flag_input':str(ROOT/'handoff/금공개 팀플/IVOL/CRSP_Monthly_1999-12_2020.csv.gz'),
 'signal_universe':'CNN-scored professor CSV, formation-month CRSP delisting filtered, deciles formed before excluding missing holding returns',
 'spread':'monthly equal-weight Q10 - Q1, before transaction costs, delisting-corrected ret_final',
 'accuracy':'monthly fraction (P(up)>=0.5) equals (ret_final>0), averaged equally over months; excludes missing outcomes',
 'last_vs_first':comparison,'limitations':'exploratory same sample, 45 months per window, no independent model selection validation; accuracy and spread use the CNN universe, not four-signal matched universe'},ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(diagnostics).to_string(index=False)); print('Spread last-first',comparison)
