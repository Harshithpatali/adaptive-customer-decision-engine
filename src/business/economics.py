from __future__ import annotations
import numpy as np
import pandas as pd

def campaign_summary(df, cost_by_action):
    rows=[]
    for action,cost in cost_by_action.items():
        s=df.loc[df.segment.eq(action)]
        n=len(s); spend=float(s.spend.sum()); conversions=int(s.conversion.sum()); visits=int(s.visit.sum())
        cost_total=n*float(cost); net=spend-cost_total
        rows.append({'action':action,'customers':n,'visits':visits,'conversions':conversions,'observed_spend':spend,'campaign_cost':cost_total,'net_observed_value':net,'revenue_per_customer':spend/n if n else 0,'conversion_rate':conversions/n if n else 0})
    return pd.DataFrame(rows)

def policy_value(actions, expected_revenue, costs):
    actions=np.asarray(actions); rev=np.asarray(expected_revenue,float)
    return float(np.mean(rev-np.array([costs[a] for a in actions])) )

def simulate_budget(policy_df, budget, costs, value_col='expected_profit'):
    x=policy_df.sort_values(value_col,ascending=False).copy(); selected=[]; spent=0.0
    for i,r in x.iterrows():
        c=float(costs.get(r['recommended_action'],0))
        if spent+c <= budget: selected.append(i); spent += c
    out=x.loc[selected].copy(); return out, {'budget':float(budget),'spent':float(spent),'customers_selected':int(len(out)),'expected_value':float(out[value_col].sum()) if len(out) else 0.0}
