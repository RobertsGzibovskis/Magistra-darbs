# config.py
# Konfigurācija
# Šo failu importē visi pārējie moduļi

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV   = os.path.join(_HERE, "data.csv")   # galvenā datu kopa
OUT   = os.path.join(_HERE, "lime_izvades_faili")
os.makedirs(OUT, exist_ok=True)

# Modeļa parametri

N_USERS      = 500    # simulēto lietotāju skaits
SVD_COMPONENTS = 40   # SVD latento iezīmju skaits
RANDOM_SEED  = 42

# LIME XAI parametri

TARGET_USER  = 1      # lietotājs, kuram tiek skaidrots ieteikums
TARGET_TRACK = -1   # dziesma, kurai tiek skaidrots ieteikums
LIME_TOP_FEATURES = 8
LIME_NUM_SAMPLES  = 2000
LIME_BACKGROUND_N = 2000   # paraugu skaits matricai


# Ieteicams: 20000
TRACK_SAMPLE_N = 20_000

# Novērtēšanas parametri

EVAL_D_SAMPLES    = 300
EVAL_D_NOISE      = 0.05
EVAL_S_TRIALS     = 7      # S metrika — perturbāciju skaits
EVAL_S_NOISE      = 0.03
EVAL_LAMBDA       = 1.0    # soda koeficients visām metrikām