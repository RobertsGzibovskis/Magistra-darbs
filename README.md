# XAI Mūzikas Ieteikumu Sistēma — LIME & SHAP Salīdzinājums

Maģistra darba praktiskā daļa. Hibrīda mūzikas ieteikumu sistēma ar LIME un SHAP skaidrojamības metodēm, novērtēta pēc četrām metrikām: Uzticamība, Sarežģītība, Noteiktība, Izturība.

---

## Projekta struktūra

```
projekts/
│
├── LIME_XAI/
│   ├── config.py              # LIME konfigurācija 
│   ├── data.py                # Datu ielāde, iezīmju matrica, lietotāju simulācija
│   ├── recommender.py         # SVD modelis + LIME explainer
│   ├── evaluation.py          # Fidelity, Simplicity, Consistency, Robustness
│   └── main.py                # LIME main skripts (viens lietotājs/dziesma)
│
├── SHAP_XAI/
│   ├── shap_config.py         # SHAP konfigurācija
│   ├── shap_data.py           # Datu ielāde
│   ├── shap_recommender.py    # SVD modelis + SHAP SamplingExplainer
│   ├── shap_evaluation.py     # Fidelity, Simplicity, Consistency, Robustness
│   └── shap_main.py           # SHAP main skripts (viens lietotājs/dziesma)
│
├── run_combined_evaluation.py # Kombinētais novērtēšanas skrējējs (LIME + SHAP)
├── data_analysis.py           # Datu analīze un vizualizācija
│
├── data.csv                   # Spotify dziesmu datu kopa
├── data_w_genres.csv          # Mākslinieku žanru dati
├── data_by_year.csv           # Vidējie audio rādītāji pa gadiem
│
├── lime_izvades_faili/        # LIME izvades (CSV, grafiki)
└── shap_izvades_faili/        # SHAP izvades (CSV, grafiki)
```

---

## Datu kopa

Projekts izmanto trīs CSV failus:

| Fails | Saturs |
|---|---|
| `data.csv` | ~170 000 dziesmu ar audio iezīmēm (energy, danceability, valence, u.c.) |
| `data_w_genres.csv` | Mākslinieku žanri |
| `data_by_year.csv` | Gada vidējie audio rādītāji |



---

## Instalācija

```bash
pip install numpy pandas scikit-learn scipy shap lime matplotlib
```

Python 3.9+

---

## Programmas palaišana

### Viens skaidrojums — LIME
```bash
cd LIME_XAI
python main.py
```
Izvada LIME skaidrojuma grafiku un metriku kopsavilkumu konsolē.

### Viens skaidrojums — SHAP
```bash
cd SHAP_XAI
python shap_main.py
```
Izvada SHAP joslu grafiku un metriku kopsavilkumu konsolē.

### Novērtēšana (metrikas)
```bash
python run_combined_evaluation.py
```
Novērtē LIME un SHAP uz identiskiem lietotājiem un dziesmām. Saglabā rezultātus CSV failos.

---

## Novērtēšanas skripta konfigurācija

`run_combined_evaluation.py` augšdaļā:

```python
EVAL_USERS      = list(range(1, 51))  # Lietotāji 1–50
EVAL_TOP_TRACKS = 3                   # Top-N dziesmas uz lietotāju
EVAL_RUNS       = 4                   # Iterāciju skaits uz instanci
```

Konsoles izvade novērtēšanas laikā:
```
Lietotāji: [1..50]  |  Dziesmas: 3  |  Izpildes: 4
  [1/600]  user=1  track=4821  run=0
  [2/600]  user=1  track=4821  run=1
  ...
Pabeigts.
```

---

## Izvades faili

Visi faili tiek saglabāti `lime_izvades_faili/`:

| Fails | Saturs |
|---|---|
| `lime_eval_res.csv` | Katra LIME instances metriku vērtības |
| `shap_eval_res.csv` | Katra SHAP instances metriku vērtības |
| `combined_comparison.csv` | LIME vs SHAP kopsavilkums  |

### `combined_comparison.csv` kolonnas

| Kolonna | Apraksts |
|---|---|
| `metric` | Metrikas nosaukums |
| `lime_mean`  | LIME vidējā vērtība un standartnovirze |
| `shap_mean`  | SHAP vidējā vērtība un standartnovirze |
| `delta_shap_minus_lime` | SHAP − LIME |

---

## Novērtēšanas metrikas

Visas četras metrikas aprēķinātas pēc vienādiem principiem abām metodēm.

**Fidelity Score** (Uzticamība) — cik precīzi skaidrojums reproducē modeļa prognozes vērtību:

Vērtība 1.0 = lēmums sakrīt, 0.0 = nesakrīt.

**Simplicity** (Sarežģītība) — iezīmju skaits, kuru relatīvā nozīme pārsniedz slieksni τ:


Mazāka vērtība = vienkāršāks skaidrojums. Aprēķināts τ ∈ {0.10, 0.05, 0.01}.

**Consistency** (Noteiktība) — divu neatkarīgu izpilžu skaidrojumu Spearman korelācija pa iezīmju ranga dziļumiem n=1..k:


Augstāka vērtība = stabilāki skaidrojumi. Robeža: >0.70 augsts.

**Robustness** (Izturība) — vidējā L1 izmaiņa atribūciju vektoros pēc mazas Gausa perturbācijas σ=0.01:


Mazāka vērtība = stabilāks skaidrojums.

---

## Ieteikumu sistēmas arhitektūra

Hibrīds modelis apvieno Collaborative Filtering (CF) un Content-Based (CB) pieejas:

```
Score(u, i) = 0.60 × CF(u, i) + 0.40 × CB(u, i) + 0.05 × popularity(i)
```

CF komponente balstās uz TruncatedSVD (40 latentās iezīmes, 500 simulēti lietotāji). CB komponente mēra kosinusa līdzību starp dziesmu un lietotāja profilu.

LIME skaidro pilno iezīmju vektoru (audio + era-dev + bināras, bez žanriem). SHAP skaidro 14 pamata audio iezīmes, izmantojot `SamplingExplainer` ar 2000 fona paraugiem.

---

## Konfigurācijas parametri

### LIME (`LIME_XAI/config.py`)

| Parametrs | Noklusējums | Apraksts |
|---|---|---|
| `N_USERS` | 500 | Simulēto lietotāju skaits |
| `SVD_COMPONENTS` | 40 | SVD latento iezīmju skaits |
| `LIME_TOP_FEATURES` | 8 | Iezīmju skaits skaidrojumā |
| `LIME_NUM_SAMPLES` | 5000 | Perturbāciju skaits LIME |
| `LIME_BACKGROUND_N` | 2000 | Fona paraugu skaits |
| `EVAL_S_TRIALS` | 7 | Robustness perturbāciju skaits |
| `TRACK_SAMPLE_N` | 20 000 | Dziesmu paraugs no datu kopas |

### SHAP (`SHAP_XAI/shap_config.py`)

| Parametrs | Noklusējums | Apraksts |
|---|---|---|
| `SHAP_BACKGROUND_N` | 2000 | Fona paraugu skaits SamplingExplainer |
| `SHAP_LOCAL_NSAMPLES` | 500 | Paraugi uz vienu SHAP izsaukumu |
| `EVAL_S_TRIALS` | 1 | Robustness perturbāciju skaits |
| `SHAP_TOP_FEATURES` | 8 | Iezīmju skaits vizualizācijā |
