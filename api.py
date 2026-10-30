from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Importation indispensable pour que le modèle puisse retrouver le Winsorizer lors du chargement
from feature_engine.outliers import Winsorizer

# Initialisation de l'application FastAPI
app = FastAPI(title="API Prédiction Diabète", version="1.0")

# Chargement du pipeline complet (qui intègre le scaler, le Winsorizer et le modèle)
pipeline = joblib.load('diabetes_model.pkl')

# Définition de la structure des données d'entrée (vos 15 features)
class PatientData(BaseModel):
    bmi: float
    age: float
    glucose: float
    blood_pressure: float
    # Ajoutez toutes vos autres features ici dans le même ordre

@app.get("/")
def home():
    return {"status": "API de prédiction du diabète opérationnelle 🚀"}

@app.post("/predict")
def predict(data: PatientData):
    # Transformation des données en tableau 2D dans l'ordre attendu
    input_features = np.array([[
        data.bmi,
        data.age,
        data.glucose,
        data.blood_pressure
    ]])

    # Le pipeline applique automatiquement le traitement puis la prédiction du modèle !
    proba = pipeline.predict_proba(input_features)[0][1]
    prediction = int(proba >= 0.401)  # Votre seuil optimisé

    return {
        "prediction": prediction,  # 1 = Risque élevé, 0 = Faible risque
        "probability": round(float(proba), 4),
        "status": "Diabétique à risque" if prediction == 1 else "Sain / Faible risque"
    }