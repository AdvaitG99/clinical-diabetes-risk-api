# Clinical Diabetes Risk Assessment API

An end-to-end machine learning microservice that predicts patient diabetic risk based on diagnostic health metrics. This project covers the full ML lifecycle i.e. from model training and threshold tuning to REST API creation and Docker containerization.

---

## What This Project Does

* **Predicts Diabetic Risk:** Accepts patient health metrics (Glucose, BMI, Age, Insulin, etc.) and computes a real-time risk score.
* **Clinical Thresholding:** Applies a custom **0.35 decision threshold** (instead of the standard 0.50) to maximize **Recall**, prioritizing sensitivity so high-risk patients are not missed.
* **Data Guardrails:** Uses Pydantic data schemas to reject invalid or impossible inputs (such as negative glucose or BMI values).
* **Containerized Deployment:** Uses FastAPI and Docker to deliver a production-ready API that can run consistently on any machine or cloud platform.

---

## How the Model Was Trained

All training, exploratory data analysis, and evaluation steps are documented in `pima_diabetes_training.ipynb`:

1. **Dataset:** Built using the PIMA Indians Diabetes Dataset (768 clinical patient records).
2. **Feature Scaling:** Applied `StandardScaler` to normalize feature distributions across different biological units (e.g., blood pressure vs. insulin levels).
3. **Model Choice:** Trained a **Logistic Regression** model for reliable baseline performance and well-calibrated probability outputs.
4. **Clinical Threshold Optimization:** In medical diagnostics, a False Negative (missing a sick patient) is far more dangerous than a False Positive (causing extra diagnostic tests). We evaluated model probabilities across multiple cutoffs and selected a **0.35 threshold** to significantly boost model Recall without degrading precision.

---

## Project Structure

* `pima_diabetes_training.ipynb` – Jupyter notebook containing EDA, preprocessing, model training, and threshold evaluation.
* `app.py` – FastAPI application handling HTTP requests and input validation.
* `predictor.py` – Inference wrapper for loading the model, scaling input features, and making predictions.
* `model.joblib` & `scaler.joblib` – Saved scikit-learn model weights and feature scaler artifacts.
* `Dockerfile` & `requirements.txt` – Container specifications and Python library dependencies.
