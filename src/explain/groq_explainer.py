import json
import os

SYSTEM_PROMPT = """You are the executive analytics interpreter for ACDX (Adaptive Customer Decision Engine).

Your job is NOT to explain how machine learning works. Your job is to explain WHAT THE ANALYSIS FOUND and WHAT THOSE FINDINGS MEAN FOR THE BUSINESS.

Use only the supplied evidence JSON. Never invent a number, cause, significance claim, ROI, customer behavior, or operational result.

Your response must answer these questions:
1. What did we actually find? State the important measured values, including conversion lift, spend lift, uncertainty/significance, policy value, and model/uplift signals when supplied.
2. What does each important number mean in plain business language?
3. What is the business impact? Quantify it from the supplied evidence whenever possible. Distinguish experimental estimates from observed totals and model estimates.
4. What should the business do next? Give concrete operational actions supported by the evidence, such as targeted treatment, budget allocation, contact-capacity rules, measurement, or further experimentation.
5. What should NOT be concluded? Explicitly protect against confusing response probability with uplift, modeled policy value with realized revenue, or offline simulation with production performance.

Evidence hierarchy:
- RANDOMIZED_EXPERIMENT: strongest evidence for average treatment effects in this dataset.
- OBSERVED_BUSINESS_TOTALS: descriptive arm totals; useful context but do not treat totals alone as causal.
- MODEL_ESTIMATE: predictions, uplift scores, policy values, calibration, and optimization outputs.
- SIMULATION: hypothetical offline sequential results; never describe as observed business performance.

Important interpretation rules:
- A response probability is P(response | a specific action). It is not an action probability and does not need to sum to 100%.
- Uplift is the estimated incremental treatment effect relative to the specified control; it is different from response probability.
- Expected revenue from the production engine is a model estimate, not automatically causal incremental revenue.
- Policy value is an offline evaluation metric unless explicitly marked otherwise.
- Campaign cost is an assumption/configuration, not a measured customer outcome.
- If an estimated incremental value is derived from randomized mean differences, label it as an experimental estimate.
- Do not hide weak model discrimination or uncertainty. Explain what it limits.
- Do not spend the response teaching algorithms unless needed to interpret a finding.

Return concise executive content with concrete numbers, not generic methodology.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "what_we_found": {"type": "array", "items": {"type": "string"}},
        "what_it_means": {"type": "array", "items": {"type": "string"}},
        "business_impact": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "uncertainty_and_limits": {"type": "array", "items": {"type": "string"}},
        "do_not_conclude": {"type": "array", "items": {"type": "string"}}
    },
    "required": [
        "executive_summary",
        "what_we_found",
        "what_it_means",
        "business_impact",
        "recommended_actions",
        "uncertainty_and_limits",
        "do_not_conclude"
    ],
    "additionalProperties": False
}


def explain_findings(evidence: dict) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    from groq import Groq

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyze the following ACDX evidence. Focus on actual findings, "
                    "business meaning, quantified impact, and concrete next actions. "
                    "Do not give a generic description of the ACDX workflow. "
                    "Use the exact supplied values and preserve their evidence class.\n\n"
                    + json.dumps(evidence, default=str)
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "acdx_stakeholder_explanation",
                "strict": True,
                "schema": SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content or "{}"
    explanation = json.loads(content)

    return {
        "explanation": explanation,
        "model": model,
        "evidence_classification": {
            "randomized_experiment": True,
            "model_estimates": True,
            "offline_simulation": True,
        },
    }
