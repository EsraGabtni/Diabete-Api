import sys
import __main__
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from feature_engine.outliers import Winsorizer

# 1. Injection indispensable pour que pickle/joblib retrouve le Winsorizer du notebook
setattr(__main__, 'Winsorizer', Winsorizer)

# Initialisation de l'application FastAPI pour MediLens
app = FastAPI(title="MediLens - API Prédiction Diabète", version="2.0")

# 2. Monter le dossier static pour charger le design (CSS, JS, images) de MediLens
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. Chargement du modèle de haute précision de ta collègue et des features
try:
    model = joblib.load("modele_haute_precision.joblib")
    with open("features_finales.json", "r", encoding="utf-8") as f:
        features_finales = json.load(f)
    SEUIL = 0.40  # Seuil de décision optimisé
    print(f"✅ Modèle chargé avec succès. {len(features_finales)} features attendues.")
except Exception as e:
    print(f"⚠️ Erreur lors du chargement : {e}")
    model = None
    features_finales = []
    SEUIL = 0.40

# Schéma Pydantic strict pour valider les données de santé
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
    # 4. Affiche directement ton interface MediLens sur la racine de ton lien Render
    return FileResponse("static/index.html")

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

        # Prédiction de la probabilité avec le modèle haute précision
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