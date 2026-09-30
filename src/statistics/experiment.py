from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2_contingency, mannwhitneyu

CONTROL = 'No E-Mail'
TREATMENTS = ['Mens E-Mail', 'Womens E-Mail']

def _prop_ci(y, n, z=1.959963984540054):
    if n == 0: return (np.nan, np.nan)
    p=y/n; den=1+z*z/n; ctr=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(ctr-half), float(ctr+half)

def _bootstrap_lift(a, b, reps=2000, seed=42):
    rng=np.random.default_rng(seed); a=np.asarray(a); b=np.asarray(b)
    if len(a)==0 or len(b)==0: return (np.nan,np.nan,np.nan)
    vals=np.empty(reps)
    for i in range(reps): vals[i]=rng.choice(a,len(a),replace=True).mean()-rng.choice(b,len(b),replace=True).mean()
    return float(np.mean(vals)), float(np.quantile(vals,.025)), float(np.quantile(vals,.975))

def compare(df, treatment, outcome, reps=2000, seed=42):
    t=df.loc[df.segment.eq(treatment), outcome].to_numpy(dtype=float)
    c=df.loc[df.segment.eq(CONTROL), outcome].to_numpy(dtype=float)
    mt,mc=t.mean(),c.mean(); diff=mt-mc; rel=diff/mc if mc else np.nan
    if outcome in ('visit','conversion'):
        yt,yc=t.sum(),c.sum(); nt,nc=len(t),len(c)
        ci_t=_prop_ci(yt,nt); ci_c=_prop_ci(yc,nc)
        table=np.array([[yt,nt-yt],[yc,nc-yc]])
        chi2,p,_,_=chi2_contingency(table, correction=False)
        boot,lo,hi=_bootstrap_lift(t,c,reps,seed)
        return {'treatment':treatment,'outcome':outcome,'n_treatment':nt,'n_control':nc,'treatment_rate':mt,'control_rate':mc,'absolute_lift':diff,'relative_lift':rel,'rate_ci_treatment':ci_t,'rate_ci_control':ci_c,'bootstrap_lift':boot,'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'p_value':float(p),'chi_square':float(chi2),'effect_significant_0_05':bool(p<.05)}
    # Continuous spend: nonparametric + bootstrap mean difference
    u,p=mannwhitneyu(t,c,alternative='two-sided')
    boot,lo,hi=_bootstrap_lift(t,c,reps,seed)
    return {'treatment':treatment,'outcome':outcome,'n_treatment':len(t),'n_control':len(c),'treatment_mean':mt,'control_mean':mc,'absolute_lift':diff,'relative_lift':rel,'bootstrap_lift':boot,'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'mann_whitney_p_value':float(p),'u_statistic':float(u)}

def run_experiment(df, reps=2000):
    rows=[]
    for tr in TREATMENTS:
        for outcome in ('visit','conversion','spend'):
            rows.append(compare(df,tr,outcome,reps=reps))
    return pd.DataFrame(rows)

def heterogeneity_table(df, subgroup_cols=('newbie','channel','history_segment','zip_code')):
    rows=[]
    for col in subgroup_cols:
        for level, g in df.groupby(col, dropna=False):
            c=g[g.segment.eq(CONTROL)]
            if len(c)<100: continue
            for tr in TREATMENTS:
                t=g[g.segment.eq(tr)]
                if len(t)<100: continue
                rows.append({'subgroup':col,'level':str(level),'treatment':tr,'n_treatment':len(t),'n_control':len(c),'treatment_conversion':t.conversion.mean(),'control_conversion':c.conversion.mean(),'absolute_conversion_lift':t.conversion.mean()-c.conversion.mean(),'treatment_spend':t.spend.mean(),'control_spend':c.spend.mean(),'absolute_spend_lift':t.spend.mean()-c.spend.mean()})
    return pd.DataFrame(rows)
