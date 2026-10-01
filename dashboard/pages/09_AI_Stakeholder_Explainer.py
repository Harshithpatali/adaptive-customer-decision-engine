import json
import requests
import streamlit as st
from ui import inject_css, hero

inject_css()
API_URL=st.sidebar.text_input('Production API URL','https://adaptive-customer-decision-engine.onrender.com').rstrip('/')
hero('AI Stakeholder Explainer','Groq-powered natural-language interpretation of ACDX evidence.',['GROQ','EVIDENCE-GROUNDED','STAKEHOLDER MODE','AUDITABLE'])
st.warning('The explainer translates supplied evidence. It cannot change model outputs, policy rules, or metrics.')
evidence_text=st.text_area('Evidence JSON',value=json.dumps({'experiment':{'population':64000},'model':{'algorithm':'XGBoost','roc_auc':0.6676941202927387,'pr_auc':0.019364064718361898},'policy':{'version':'value_max_v1'}},indent=2),height=280)
if st.button('Explain findings',type='primary'):
    try:
        evidence=json.loads(evidence_text)
        r=requests.post(API_URL+'/explain',json={'evidence':evidence},timeout=45)
        r.raise_for_status()
        result=r.json()
        st.markdown('### Stakeholder interpretation')
        st.write(result['explanation'])
        st.caption('Model: '+result.get('model','configured Groq model'))
    except Exception as exc: st.error(f'Explainer unavailable: {exc}')
st.markdown('### Intended use')
st.write('Use this layer to translate statistical and model outputs into business language while preserving the distinction between experimental evidence, model estimates, and offline simulations.')