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
    prediction = predict_claim(data)
    rules = retrieve_rules()
    
    prompt = f"""
    You are an insurance fraud investigator.
    
    Model Prediction:
    Fraud Probability: {prediction['fraud_probability']}
    
    Fraud Knowledge Base:
    {rules}
    
    Claim Data:
    {data}
    
    Generate:
    1. Risk Assessment
    2. Key Suspicious Factors
    3. Recommended Action
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You're a intelligent fraud detection agent. Response only in well formatted Markdown without code blocks."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
    
    