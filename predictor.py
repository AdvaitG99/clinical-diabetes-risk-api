import joblib
import pandas as pd

class ClinicalPredictor:
    def __init__(self, scaler_path='scaler.joblib', model_path='model.joblib', threshold=0.35):
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)
        self.threshold = threshold
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 
            'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Age'
        ]

    def predict_patient_risk(self, patient_dict: dict) -> dict:
        df_patient = pd.DataFrame([patient_dict])[self.feature_names]
        scaled_array = self.scaler.transform(df_patient)
        scaled_features = pd.DataFrame(scaled_array, columns=self.feature_names)
        
        risk_probability = float(self.model.predict_proba(scaled_features)[:, 1][0])
        is_high_risk = risk_probability >= self.threshold
        
        return {
            "diabetic_risk_score": round(risk_probability, 4),
            "custom_threshold_applied": self.threshold,
            "high_risk_flag": int(is_high_risk),
            "clinical_recommendation": (
                "Flagged for immediate confirmatory diagnostic testing (HbA1c)."
                if is_high_risk else
                "Low risk detected; maintain standard periodic screening schedule."
            )
        }
