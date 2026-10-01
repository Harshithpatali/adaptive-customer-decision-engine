import json
import os

SYSTEM_PROMPT = """You are the stakeholder explanation layer for ACDX, an enterprise customer decision intelligence system.
Explain analytics to a non-technical business stakeholder. Never invent metrics, causes, significance, customer behavior, or ROI.
Only use facts supplied in the evidence JSON. Distinguish observed randomized experimental evidence, model estimates, and offline policy simulations.
Explain uncertainty when confidence intervals or model limitations are present. Never turn model predictions into causal claims.
Do not make the decision yourself; explain what the decision engine calculated and why."""

def explain_findings(evidence: dict) -> dict:
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key: raise RuntimeError('GROQ_API_KEY is not configured.')
    from groq import Groq
    model = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(model=model, temperature=0.2, messages=[
        {'role':'system','content':SYSTEM_PROMPT},
        {'role':'user','content':'Explain these ACDX findings for a stakeholder. Keep it concise, structured, and evidence-first.\n\n' + json.dumps(evidence, default=str)}
    ])
    return {'explanation': response.choices[0].message.content or '', 'model': model}