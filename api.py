import sys
import __main__
from feature_engine.outliers import Winsorizer

# Injection indispensable pour que pickle retrouve le Winsorizer enregistré depuis le notebook
setattr(__main__, 'Winsorizer', Winsorizer)

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialisation de l'application FastAPI
app = FastAPI(title="API Prédiction Diabète", version="1.0")

# Chargement du pipeline complet
pipeline = joblib.load('diabetes_model.pkl')

# Définition de la structure des données d'entrée (vos features)
class PatientData(BaseModel):
    bmi: float
    age: float
    glucose: float
    blood_pressure: float
    # Ajoutez vos autres features ici dans le même ordre

@app.get("/")
def home():
    return {"status": "API de prédiction du diabète opérationnelle 🚀"}

@app.post("/predict")
def predict(data: PatientData):
    input_features = np.array([[
        data.bmi,
        data.age,
        data.glucose,
        data.blood_pressure
    ]])

    proba = pipeline.predict_proba(input_features)[0][1]
    prediction = int(proba >= 0.401)  # Seuil optimisé

    return {
        "prediction": prediction,
        "probability": round(float(proba), 4),
        "status": "Diabétique à risque" if prediction == 1 else "Sain / Faible risque"
    }