from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd

app = FastAPI(title="API Backend Prédiction Diabète")

# Chargement du modèle et des features
try:
    model = joblib.load("modele_haute_precision.joblib")
    with open("features_finales.json", "r", encoding="utf-8") as f:
        features_finales = json.load(f)
    SEUIL = 0.40  # Seuil de décision
    print(f"✅ Modèle chargé avec succès. {len(features_finales)} features attendues.")
except Exception as e:
    print(f"⚠️ Erreur lors du chargement : {e}")
    model = None
    features_finales = []
    SEUIL = 0.40

# Schéma Pydantic strict pour valider les données envoyées par le front de tes amis
class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120)
    bmi: float = Field(..., ge=10.0, le=70.0)
    systolic_bp: float = Field(..., ge=60.0, le=250.0)
    diastolic_bp: float = Field(..., ge=30.0, le=150.0)
    heart_rate: float = Field(..., ge=30.0, le=200.0)
    hypertension_status: str = Field("No")
    family_history_diabetes: str = Field("No")

@app.get("/")
def read_root():
    return {"message": "API Backend Diabète active et prête pour intégration 🚀"}

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé sur le serveur.")
    
    try:
        input_dict = data.dict()
        
        full_input = {}
        for col in features_finales:
            if col in input_dict:
                val = input_dict[col]
                if col in ['hypertension_status', 'family_history_diabetes']:
                    full_input[col] = str(val).capitalize()
                else:
                    full_input[col] = float(val)
            else:
                full_input[col] = 0.0 

        input_df = pd.DataFrame([full_input])[features_finales]

        # Prédiction de la probabilité
        proba = float(model.predict_proba(input_df)[0, 1])
        prediction = int(proba >= SEUIL)

        return {
            "statut_predit": prediction,
            "libelle": "Alerte Diabète" if prediction == 1 else "Non diabétique (Sain)",
            "probabilite": round(proba, 4),
            "seuil_utilise": SEUIL
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de traitement : {str(e)}")