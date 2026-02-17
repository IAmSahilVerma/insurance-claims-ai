from ml.predict import predict_claim
from rag.retriever import retrieve_rules
from dotenv import load_dotenv
from openai import OpenAI
import os
import json

load_dotenv(override=True)
api_key=os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def investigate_claim(data: dict):
    # Get model prediction (includes SHAP key_risk_factors)
    prediction = predict_claim(data)

    # Retrieve RAG rules
    rules = retrieve_rules(claim_data=data, shap_factors=prediction["key_risk_factors"])

    prompt = f"""
    You are an expert insurance fraud investigator AI.

    You MUST return a valid JSON object only.
    Do NOT return markdown.
    Do NOT include explanations outside JSON.

    Use the information below to generate your assessment.

    ========================
    MODEL OUTPUT
    ========================
    Fraud Probability: "{prediction['fraud_probability']}"
    Risk Level: "{prediction['risk_level']}"
    Top SHAP Risk Factors: "{prediction['key_risk_factors']}"

    ========================
    FRAUD KNOWLEDGE BASE
    ========================
    "{rules}"

    ========================
    CLAIM DATA
    ========================
    "{data}"

    ========================
    OUTPUT FORMAT (STRICT JSON)
    ========================
    {{
    "risk_level": "Low | Medium | High",
    "fraud_probability": 0.0,
    "key_risk_factors": ["string"],
    "justification": "clear reasoning using model + SHAP + rules",
    "recommended_action": "Approve | Manual Review | Escalate Investigation | Reject"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,  # Lower = more deterministic
        response_format={"type": "json_object"},  # Forces JSON output
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict fraud detection AI. "
                    "You MUST return valid JSON only. "
                    "No markdown. No explanations outside JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(response.choices[0].message.content)