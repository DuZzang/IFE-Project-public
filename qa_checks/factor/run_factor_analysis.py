"""Factor regressions for saved 2005-2019 IFE long-only monthly portfolios.

No portfolio construction or training is performed. Official French MOM and the
project's own WML are separate regressors; they are not interchangeable.
"""
from pathlib import Path
import csv
import argparse
import hashlib
import json
import re
import zipfile
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
parser=argparse.ArgumentParser(description='Reproduce the factor regression with packaged data')
parser.add_argument('--project-root', type=Path, default=SCRIPT.parents[2], help='Extracted project root')
parser.add_argument('--factor-dir', type=Path, default=SCRIPT.parents[1] / 'factors', help='Directory containing official 202609 French ZIPs')
parser.add_argument('--output-dir', type=Path, default=SCRIPT.parent / 'results_rerun', help='Directory for newly generated output')
parser.add_argument('--official-vintage', action='store_true', default=True, help='Use 202609 FF3/FF5 and French MOM (default)')
args=parser.parse_args()
ROOT = args.project_root.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
RET = ROOT / 'outputs/longonly_d10_vs_sp500/longonly_d10_monthly_net.csv'
SUMMARY = ROOT / 'outputs/longonly_d10_vs_sp500/longonly_d10_vs_sp500.csv'
FFDIR = ROOT / 'handoff/금공개 팀플/IVOL'
FF3 = FFDIR / 'F-F_Research_Data_Factors_CSV.zip'
FF5 = FFDIR / 'F-F_Research_Data_5_Factors_2x3_CSV.zip'
WML = ROOT / 'outputs/managed_portfolios/bsc_wml_monthly.csv'
MOM = args.factor_dir.resolve() / 'F-F_Momentum_Factor_CSV.zip'

def hash_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_ff(p):
    with zipfile.ZipFile(p) as z:
        text = z.read(z.namelist()[0]).decode('utf-8-sig', errors='replace')
    rows = list(csv.reader(text.splitlines()))
    header = next(i for i,r in enumerate(rows) if len(r)>2 and r[0].strip()=='' and r[1].strip().lower()=='mkt-rf')
    cols = [x.strip().lower().replace('-', '_') for x in rows[header][1:]]
    selected=[]
    for r in rows[header+1:]:
        if len(r)>len(cols) and re.fullmatch(r'\d{6}', r[0].strip()):
            selected.append([r[0].strip()[:4]+'-'+r[0].strip()[4:],*[float(x.strip())/100 for x in r[1:len(cols)+1]]])
    f = pd.DataFrame(selected,columns=['month',*cols]).set_index('month')
    if not f.index.is_unique: raise ValueError(f'duplicate factor months in {p}')
    return f, text.splitlines()[0]

def load_mom(p):
    with zipfile.ZipFile(p) as z:
        text=z.read(z.namelist()[0]).decode('utf-8-sig',errors='replace')
    rows=list(csv.reader(text.splitlines()))
    header=next(i for i,r in enumerate(rows) if len(r)>1 and r[0].strip()=='' and r[1].strip().lower()=='mom')
    selected=[]
    for r in rows[header+1:]:
        if len(r)>1 and re.fullmatch(r'\d{6}',r[0].strip()):
            value=float(r[1].strip())
            selected.append([r[0].strip()[:4]+'-'+r[0].strip()[4:],np.nan if value in (-99.99,-999.0) else value/100])
    f=pd.DataFrame(selected,columns=['month','mom']).set_index('month')
    if not f.index.is_unique: raise ValueError('duplicate momentum months')
    return f,text.splitlines()[0]

def regression(y, x, lags=4):
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, dtype=float)])
    if not np.isfinite(y).all() or not np.isfinite(X).all(): raise ValueError('nonfinite inputs')
    if np.linalg.matrix_rank(X)!=X.shape[1]: raise ValueError('singular design')
    coef = np.linalg.lstsq(X,y,rcond=None)[0]
    residual = y-X@coef
    bread = np.linalg.inv(X.T@X)
    scores=X*residual[:,None]
    meat=scores.T@scores
    for lag in range(1,lags+1):
        cross=scores[lag:].T@scores[:-lag]
        meat += (1-lag/(lags+1))*(cross+cross.T)
    se=np.sqrt(np.maximum(np.diag(bread@meat@bread),0))
    r2=1-np.sum(residual**2)/np.sum((y-y.mean())**2)
    return coef,se,r2

def main():
    ff3_path=args.factor_dir.resolve()/'official_F-F_Research_Data_Factors_CSV.zip' if args.official_vintage else FF3
    ff5_path=args.factor_dir.resolve()/'official_F-F_Research_Data_5_Factors_2x3_CSV.zip' if args.official_vintage else FF5
    suffix='_official_202609' if args.official_vintage else ''
    r=pd.read_csv(RET,dtype={'holding_month':str}).set_index('holding_month')
    expected=pd.period_range('2005-01','2019-12',freq='M').astype(str)
    r=r.loc[(r.index>='2005-01')&(r.index<='2019-12')]
    assert r.index.is_unique and list(r.index)==list(expected), 'returns months mismatch'
    f3,v3=load_ff(ff3_path); f5,v5=load_ff(ff5_path); mom,vm=load_mom(MOM)
    w=pd.read_csv(WML,dtype={'holding_month':str}).set_index('holding_month')
    assert w.index.is_unique
    f3=f3.reindex(expected); f5=f5.reindex(expected); mom=mom.reindex(expected); w=w.reindex(expected)
    assert f3[['mkt_rf','smb','hml','rf']].notna().all().all()
    assert f5[['mkt_rf','smb','hml','rmw','cma','rf']].notna().all().all()
    assert mom['mom'].notna().all()
    assert w[['WML EW','WML VW']].notna().all().all()
    assert r.notna().all().all()
    rf_maxdiff=float((f3['rf']-f5['rf']).abs().max())
    assert rf_maxdiff<1e-12, 'FF3 vs FF5 RF differs'
    names={'Top50 CNN70 EW':'CNN 70% | top50 EW','Top50 Equal25 EW':'Equal 25% | top50 EW','Top50 CNN-only EW':'CNN only (matched) | top50 EW','S&P 500 TR':'sp500tr'}
    frames={
        'SP500_CAPM': pd.DataFrame({'sp_excess':r['sp500tr']-f5['rf']},index=expected),
        'FF3':f3[['mkt_rf','smb','hml']],
        'FF5':f5[['mkt_rf','smb','hml','rmw','cma']],
        'FF3_plus_French_MOM':f3[['mkt_rf','smb','hml']].assign(mom=mom['mom']),
        'FF5_plus_French_MOM':f5[['mkt_rf','smb','hml','rmw','cma']].assign(mom=mom['mom']),
        'FF3_plus_project_WML_VW':f3[['mkt_rf','smb','hml']].assign(project_wml_vw=w['WML VW']),
        'FF5_plus_project_WML_VW':f5[['mkt_rf','smb','hml','rmw','cma']].assign(project_wml_vw=w['WML VW']),
    }
    aligned=pd.DataFrame(index=expected)
    for label,col in names.items(): aligned[label]=r[col]
    for col in f3.columns: aligned['ff3_'+col]=f3[col]
    for col in f5.columns: aligned['ff5_'+col]=f5[col]
    aligned['french_mom']=mom['mom']
    aligned['project_wml_vw']=w['WML VW']
    aligned.index.name='holding_month'
    aligned.to_csv(OUT/f'aligned_monthly{suffix}.csv')
    result=[]; loadings=[]
    for label,col in names.items():
        for model,x in frames.items():
            if label=='S&P 500 TR' and model=='SP500_CAPM': continue
            coef,se,r2=regression(r[col]-f5['rf'],x)
            result.append({'portfolio':label,'model':model,'months':len(r),'annualized_alpha':12*coef[0],'alpha_monthly':coef[0],'alpha_t_nw4':coef[0]/se[0],'alpha_se_monthly':se[0],'r_squared':r2})
            for factor,b,s in zip(['intercept',*x.columns],coef,se):
                loadings.append({'portfolio':label,'model':model,'factor':factor,'beta':b,'se_nw4':s,'t_nw4':b/s})
    for comparator in ['Top50 CNN-only EW','Top50 Equal25 EW']:
        diff=r[names['Top50 CNN70 EW']]-r[names[comparator]]
        for model in ['FF3','FF5','FF3_plus_French_MOM','FF5_plus_French_MOM','FF5_plus_project_WML_VW']:
            x=frames[model]; coef,se,r2=regression(diff,x)
            result.append({'portfolio':f'Top50 CNN70 EW minus {comparator}','model':model,'months':len(r),'annualized_alpha':12*coef[0],'alpha_monthly':coef[0],'alpha_t_nw4':coef[0]/se[0],'alpha_se_monthly':se[0],'r_squared':r2})
    result=pd.DataFrame(result); result.to_csv(OUT/f'factor_alphas{suffix}.csv',index=False)
    pd.DataFrame(loadings).to_csv(OUT/f'factor_loadings{suffix}.csv',index=False)
    check=pd.read_csv(SUMMARY)
    row=check[(check['portfolio']=='CNN 70%')&(check['weighting']=='top50 EW')&(check['returns']=='net 0.10%')]
    if len(row)!=1: raise ValueError('could not identify official CAPM row')
    capm=result[(result['portfolio']=='Top50 CNN70 EW')&(result['model']=='SP500_CAPM')].iloc[0]
    delta=float(capm['annualized_alpha']-row.iloc[0]['alpha_sp'])
    if not args.official_vintage: assert abs(delta)<1e-10, f'CAPM alpha mismatch {delta}'
    older3,_=load_ff(FF3); older5,_=load_ff(FF5)
    vintage_maxdiff={'ff3_'+c:float((f3[c]-older3.reindex(expected)[c]).abs().max()) for c in ['mkt_rf','smb','hml','rf']}
    vintage_maxdiff.update({'ff5_'+c:float((f5[c]-older5.reindex(expected)[c]).abs().max()) for c in ['mkt_rf','smb','hml','rmw','cma','rf']})
    meta={'months':len(r),'period':['2005-01','2019-12'],'factor_vintages':[v3,v5,vm],'factor_percent_converted_to_decimals':True,'risk_free':'FF monthly RF; FF3 and FF5 agree','rf_max_abs_difference':rf_maxdiff,'NW_lags':4,'NW_kernel':'Bartlett; no small sample correction','stored_capm_alpha_diff':delta,'factor_max_monthly_diff_vs_202607':vintage_maxdiff,'source_sha256':{p.name:hash_file(p) for p in [RET,SUMMARY,ff3_path,ff5_path,MOM,WML]},'factor_vintage_warning':'All factors 202609 CRSP vintage.' if args.official_vintage else 'FF3/FF5 use 202607 CRSP vintage; French MOM uses 202609 CRSP vintage. Mixed-vintage specification is provisional.','project_wml_warning':'Project WML VW is own decile 10-1 momentum strategy, not French MOM, and may be mechanically related to evaluated strategy.','selection_warning':'CNN70 selected on same 2005-2019 period, not independent confirmation.'}
    (OUT/f'metadata{suffix}.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print(result.to_string(index=False)); print(json.dumps(meta,ensure_ascii=False))

if __name__=='__main__': main()
