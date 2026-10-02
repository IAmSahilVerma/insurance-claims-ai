from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from agent.fraud_agent import Investigation, investigate_claim
from ml.predict import predict_claim

EXAMPLE_CLAIM = {
    "Month": "Sep",
    "WeekOfMonth": "3",
    "DayOfWeek": "Saturday",
    "Make": "Ford",
    "AccidentArea": "Urban",
    "DayOfWeekClaimed": "Sunday",
    "MonthClaimed": "Sep",
    "WeekOfMonthClaimed": "3",
    "Sex": "Male",
    "MaritalStatus": "Married",
    "Age": 34,
    "Fault": "Policy Holder",
    "PolicyType": "Utility - All Perils",
    "VehicleCategory": "Utility",
    "VehiclePrice": "30000 to 39000",
    "Deductible": 400,
    "DriverRating": 2,
    "Days_Policy_Accident": "15 to 30",
    "Days_Policy_Claim": "15 to 30",
    "PastNumberOfClaims": "1",
    "AgeOfVehicle": "3 years",
    "AgeOfPolicyHolder": "31 to 40",
    "PoliceReportFiled": "Yes",
    "WitnessPresent": "No",
    "AgentType": "External",
    "NumberOfSuppliments": "1 to 2",
    "NumberOfCars": "2 vehicles",
    "BasePolicy": "All Perils",
}


class Claim(BaseModel):
    """An insurance claim, using the same fields the model was trained on."""
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_CLAIM]})

    Month: str
    WeekOfMonth: str
    DayOfWeek: str
    Make: str
    AccidentArea: str
    DayOfWeekClaimed: str
    MonthClaimed: str
    WeekOfMonthClaimed: str
    Sex: str
    MaritalStatus: str
    Age: int
    Fault: str
    PolicyType: str
    VehicleCategory: str
    VehiclePrice: str
    Deductible: int
    DriverRating: int
    Days_Policy_Accident: str
    Days_Policy_Claim: str
    PastNumberOfClaims: str
    AgeOfVehicle: str
    AgeOfPolicyHolder: str
    PoliceReportFiled: str
    WitnessPresent: str
    AgentType: str
    NumberOfSuppliments: str
    NumberOfCars: str
    BasePolicy: str


class Prediction(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    fraud_probability: float
    prediction: int
    key_risk_factors: list[str]


app = FastAPI(
    title="Insurance Claims AI",
    description="Fraud scoring with LightGBM and SHAP, rule retrieval, and LLM-written justifications.",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(claim: Claim):
    """Model only: fraud probability, risk level and top SHAP factors. No LLM call."""
    return predict_claim(claim.model_dump())


@app.post("/investigate", response_model=Investigation)
def investigate(claim: Claim):
    """Full pipeline: model, SHAP, rule retrieval, routing and an LLM justification."""
    return investigate_claim(claim.model_dump())
