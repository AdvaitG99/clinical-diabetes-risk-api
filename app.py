from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predictor import ClinicalPredictor

class PatientData(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of times pregnant", examples=[3])
    Glucose: float = Field(..., ge=0.0, description="Plasma glucose concentration (mg/dL)", examples=[168.0])
    BloodPressure: float = Field(..., ge=0.0, description="Diastolic blood pressure (mm Hg)", examples=[74.0])
    SkinThickness: float = Field(..., ge=0.0, description="Triceps skin fold thickness (mm)", examples=[32.0])
    Insulin: float = Field(..., ge=0.0, description="2-Hour serum insulin (mu U/ml)", examples=[180.0])
    BMI: float = Field(..., ge=0.0, description="Body mass index", examples=[37.5])
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, description="Diabetes pedigree function score", examples=[0.627])
    Age: int = Field(..., ge=0, description="Age in years", examples=[42])

class PredictionResponse(BaseModel):
    diabetic_risk_score: float
    custom_threshold_applied: float
    high_risk_flag: int
    clinical_recommendation: str

app = FastAPI(
    title="Clinical Diabetes Risk Assessment API",
    description="Microservice for real-time diabetic risk scoring with custom clinical thresholding (0.35).",
    version="1.0.0"
)

try:
    predictor = ClinicalPredictor()
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "healthy", "service": "Clinical Diabetes Risk Assessment API"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_risk(patient: PatientData):
    try:
        patient_dict = patient.model_dump()
        return predictor.predict_patient_risk(patient_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
