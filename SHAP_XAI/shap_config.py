# shap_config.py
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV   = os.path.join(_HERE, "C:\Magistrs_praktiskais\LIME_XAI\data.csv")
OUT   = os.path.join(_HERE, "shap_izvades_faili")
os.makedirs(OUT, exist_ok=True)

# Modeļa parametri
N_USERS        = 500
SVD_COMPONENTS = 40
RANDOM_SEED    = 42
TRACK_SAMPLE_N = 20_000

# SHAP parametri
TARGET_USER         = 1
TARGET_TRACK        = 18696
SHAP_BACKGROUND_N   = 2000    # fona paraugu skaits SamplingExplainer
SHAP_TOP_FEATURES   = 8
SHAP_LOCAL_NSAMPLES = 5000   # Paraugi uz vienu SHAP izsaukumu

# Novērtēšanas parametri
EVAL_S_TRIALS = 7            # Perturbāciju skaits
