# config.py
# Konfigurācija
# Šo failu importē visi pārējie moduļi

import os
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV   = os.path.join(_HERE, "spotify_2015_2025_85k.csv")   # datu kopa
OUT   = os.path.join(_HERE, "lime_izvades_faili")           # grafiku mape
os.makedirs(OUT, exist_ok=True)

# Modeļa parametri

N_USERS      = 500    # simulēto lietotāju skaits
SVD_COMPONENTS = 40   # SVD latento iezīmju skaits
RANDOM_SEED  = 42

# LIME XAI parametri

TARGET_USER  = 7      # lietotājs, kuram tiek skaidrots ieteikums
TARGET_TRACK = 1042   # dziesma, kurai tiek skaidrots ieteikums
LIME_TOP_FEATURES = 15
LIME_NUM_SAMPLES  = 2000
LIME_BACKGROUND_N = 2000   # paraugu skaits matricai

# Novērtēšanas parametri

EVAL_DELTA        = 0.05   # D metrika — tolerance
EVAL_D_SAMPLES    = 300
EVAL_D_NOISE      = 0.05
EVAL_R_THRESHOLD  = 5      # R metrika — max pieļaujamais noteikumu skaits
EVAL_F_THRESHOLD  = 5      # F metrika — max pieļaujamais iezīmju skaits
EVAL_S_TRIALS     = 7      # S metrika — perturbāciju skaits
EVAL_S_NOISE      = 0.03
EVAL_LAMBDA       = 1.0    # soda koeficients visām metrikām