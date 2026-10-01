from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from ui import inject_css, hero, metric_card

inject_css()
hero('Decision Optimization','Constrained multi-treatment allocation: maximize modeled business value under budget and contact capacity.',['MILP','BUDGET','CAPACITY','MULTI-TREATMENT'])
st.info('Research-time optimizer using frozen customer-level scores. It does not retrain models.')
uploaded = st.file_uploader('Upload customer action scores', type=['csv'])
budget = st.number_input('Campaign budget', min_value=0.0, value=1000.0, step=50.0)
capacity = st.number_input('Maximum contacted customers', min_value=0, value=1000, step=50)
if uploaded:
    from src.policy.optimizer import OptimizationConstraints, optimize_customer_policy, summarize_policy
    scores = pd.read_csv(uploaded)
    try:
        selected = optimize_customer_policy(scores, OptimizationConstraints(budget=budget, contact_capacity=int(capacity)))
        summary = summarize_policy(selected)
        a,b,c,d=st.columns(4)
        with a: metric_card('Customers',f"{summary['customers']:,}")
        with b: metric_card('Contacts',f"{summary['contacts']:,}",f'Cap {capacity:,}')
        with c: metric_card('Campaign spend',f"${summary['spend']:,.2f}",f'Budget ${budget:,.2f}')
        with d: metric_card('Net expected value',f"${summary['net_expected_value']:,.2f}")
        chart=pd.DataFrame({'Action':list(summary['action_counts'].keys()),'Customers':list(summary['action_counts'].values())})
        fig=px.bar(chart,x='Action',y='Customers',color='Action',text_auto=True,title='Optimized treatment allocation',template='plotly_dark')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})
        st.dataframe(selected,use_container_width=True,hide_index=True)
    except Exception as exc: st.error(f'Optimization failed: {exc}')
else:
    st.markdown('''### Optimization contract

**Objective:** maximize total expected value net of campaign cost.

**Constraints:** each customer gets exactly one action; total campaign spend cannot exceed budget; contacted customers cannot exceed capacity.

Because the problem is a binary assignment problem with explicit resource constraints, ACDX solves it as a mixed-integer linear program rather than applying an arbitrary score threshold.
''')
st.caption('Expected value remains model-dependent. The optimizer allocates according to those frozen estimates; it does not create causal evidence by itself.')