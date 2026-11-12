"""
Script de re-export du modèle.
Re-sauvegarde le .joblib pour que Winsorizer soit associé
au module 'winsorizer' plutôt qu'à '__main__'.
"""
import sys
import os
import pickle
import joblib

# Importer Winsorizer depuis le module dédié
from winsorizer import Winsorizer

# Patcher __main__ pour que pickle retrouve la classe pendant le chargement
sys.modules['__main__'].Winsorizer = Winsorizer

# Charger l'ancien modèle (pickle va maintenant trouver Winsorizer)
modele = joblib.load("modele_haute_precision.joblib")

# Re-sauvegarder — maintenant Winsorizer sera associé au module 'winsorizer'
# On doit remplacer la classe dans le pipeline
def patch_winsorizer(obj):
    """Parcourt récursivement le pipeline et remplace les instances de Winsorizer"""
    if hasattr(obj, 'transformers_'):
        for i, (name, transformer, cols) in enumerate(obj.transformers_):
            patch_winsorizer(transformer)
    if hasattr(obj, 'steps'):
        for i, (name, step) in enumerate(obj.steps):
            patch_winsorizer(step)
    if type(obj).__name__ == 'Winsorizer':
        obj.__class__ = Winsorizer

patch_winsorizer(modele)

joblib.dump(modele, "modele_haute_precision.joblib")
print("Modele re-exporte avec succes !")

# Vérification
modele2 = joblib.load("modele_haute_precision.joblib")
print("Verification : chargement OK !")
print(type(modele2))
