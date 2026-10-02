from ml.predict import predict_claim
from rag.retriever import retrieve_rules
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
import os

load_dotenv(override=True)

_client = None


def get_client() -> OpenAI:
    """Create the OpenAI client on first use, so the API can start without a key."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

# Routing thresholds on the model's fraud probability.
# Below AUTO_APPROVE_BELOW, claims are approved automatically.
# Everything else goes to a person. Nothing is ever rejected automatically.
AUTO_APPROVE_BELOW = 0.3
ESCALATE_AT = 0.7


class LLMReply(BaseModel):
    """What the LLM is allowed to return: an explanation, nothing else."""
    justification: str = Field(min_length=1)


class Investigation(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    fraud_probability: float
    key_risk_factors: list[str]
    justification: str
    recommended_action: Literal["Approve", "Manual Review", "Escalate Investigation"]
    needs_human_review: bool


def route_claim(fraud_probability: float) -> tuple[str, bool]:
    """Decide the action from the model's probability, not from the LLM."""
    if fraud_probability < AUTO_APPROVE_BELOW:
        return "Approve", False
    if fraud_probability < ESCALATE_AT:
        return "Manual Review", True
    return "Escalate Investigation", True


def generate_justification(prediction: dict, rules: list, data: dict, action: str) -> str:
    """Ask the LLM to explain a decision that has already been made."""
    prompt = f"""
    You are assisting an insurance fraud investigator.

    The routing decision has already been made: {action}.
    Do not change the decision, the probability or the risk level.
    Write a short justification (2 to 3 sentences) for the investigator,
    using the model output, the SHAP risk factors and the fraud rules below.

    MODEL OUTPUT
    Fraud probability: {prediction['fraud_probability']:.4f}
    Risk level: {prediction['risk_level']}
    Top SHAP risk factors: {prediction['key_risk_factors']}

    FRAUD KNOWLEDGE BASE
    {rules}

    CLAIM DATA
    {data}

    Return only this JSON object:
    {{"justification": "your explanation"}}
    """

    response = get_client().chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You explain fraud model decisions. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    reply = LLMReply.model_validate_json(response.choices[0].message.content)
    return reply.justification


def investigate_claim(data: dict) -> dict:
    prediction = predict_claim(data)
    rules = retrieve_rules(claim_data=data, shap_factors=prediction["key_risk_factors"])
    action, needs_human_review = route_claim(prediction["fraud_probability"])

    try:
        justification = generate_justification(prediction, rules, data, action)
    except (OpenAIError, ValidationError) as e:
        # If the explanation can't be produced or doesn't match the schema,
        # a person looks at the claim instead of trusting a partial result.
        justification = f"Automated explanation unavailable ({type(e).__name__}). Sent for manual review."
        action, needs_human_review = "Manual Review", True

    result = Investigation(
        risk_level=prediction["risk_level"],
        fraud_probability=prediction["fraud_probability"],
        key_risk_factors=prediction["key_risk_factors"],
        justification=justification,
        recommended_action=action,
        needs_human_review=needs_human_review,
    )
    return result.model_dump()
