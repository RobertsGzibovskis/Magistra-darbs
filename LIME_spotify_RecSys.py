# Programmas darbības princips
#1. Ielādē un apstrādā datus
#2. Izveido vienumu-īpašību matricu (audio + metadati + OHE)
#3. Simulē lietotājus un to preferences, lietotāju profilus
#4. Tiek izmantoa hibrīdā ieteikumu sistēma, kas sniedz prognozēto vērtējumu, izmantojot SVD (uz sadarbību balstīta metode) un kosinusa līdzība
#5. Izmantots "LIME tabular explainer", gan vienam ieteikumam (lokāli), gan globāli izskaidro sistēmas ieteikumus


import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

import lime
import lime.lime_tabular

# Definētas krāsas priekš grafikiem
DARK_BG  = "#0f1117"
DARK_AX  = "#1a1d27"
GREEN    = "#1db954"
RED      = "#e05c5c"
PURPLE   = "#9b59b6"
YELLOW   = "#f5c842"
TEXT_COL = "#dddddd"
GRID_COL = "#2a2d3a"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": DARK_AX,
    "axes.edgecolor":   GRID_COL,"axes.labelcolor": TEXT_COL,
    "xtick.color":      TEXT_COL,"ytick.color":    TEXT_COL,
    "text.color":       TEXT_COL,"grid.color":     GRID_COL,
    "legend.facecolor": DARK_AX, "legend.edgecolor": GRID_COL,
})


# 1.  Tiek ielādēti un apstrādāti dati


_HERE = os.path.dirname(os.path.abspath(__file__))
CSV   = os.path.join(_HERE, "spotify_2015_2025_85k.csv")  # CSV jāatrodas tajā pašā direktorijā, kur programmas fails
OUT   = os.path.join(_HERE, "lime_izvades_faili")               # Grafiku saglabāšana
os.makedirs(OUT, exist_ok=True)
print(f"Izvades direktorija: {OUT}")
# ─────────────────────────────────────────────────────────────────────────────
print("Datukopa")
df = pd.read_csv(CSV).dropna().reset_index(drop=True)
print(f"  Rindas: {len(df):,}  |  Kolonnas: {df.columns.tolist()}")

# Dziesmas izlaišanas gads
df["release_year"] = pd.to_datetime(df["release_date"]).dt.year

df["engagement"] = np.log1p(df["stream_count"])
df["engagement"] = (df["engagement"] - df["engagement"].min()) / \
                   (df["engagement"].max() - df["engagement"].min())

# Laiks tiek pārvērsts minūtēs (no ms, lai būtu vieglāk uztvert datus)
df["duration_min"] = df["duration_ms"] / 60_000

N_TRACKS = len(df)
GENRES   = df["genre"].unique().tolist()
COUNTRIES= df["country"].unique().tolist()

print(f"  Žanri  : {sorted(GENRES)}")
print(f"  Valstis: {len(COUNTRIES)}")

# 2.  Vienumu īpašību matrica

# Dziesmu metadati, skaitliski
CONTINUOUS = [
    "danceability", "energy", "loudness", "tempo",
    "instrumentalness", "popularity", "duration_min",
    "engagement", "release_year"
]

# Bināri dati
BINARY = ["explicit", "mode"]

# OHE (One-Hot Encoding) kategorijas
genre_ohe   = pd.get_dummies(df["genre"],   prefix="g")
country_ohe = pd.get_dummies(df["country"], prefix="c")
label_ohe   = pd.get_dummies(df["label"],   prefix="lbl")

# Dziesmas tonalitāte apstrādāta ar OHE (0-11 tonalitātes)
key_ohe = pd.get_dummies(df["key"], prefix="key")

raw_features = pd.concat([
    df[CONTINUOUS].reset_index(drop=True),
    df[BINARY].reset_index(drop=True),
    genre_ohe.reset_index(drop=True),
    country_ohe.reset_index(drop=True),
    label_ohe.reset_index(drop=True),
    key_ohe.reset_index(drop=True),
], axis=1).astype(float)

FEATURE_NAMES = raw_features.columns.tolist()

scaler         = MinMaxScaler()
feature_matrix = scaler.fit_transform(raw_features)   # (N_TRACKS, n_feats)

print(f"\nVienumu Matrica: {feature_matrix.shape}")
print(f"  Continuous : {CONTINUOUS}")
print(f"  Binary     : {BINARY}")
print(f"  OHE dims   : genre={len(genre_ohe.columns)} "
      f"country={len(country_ohe.columns)} "
      f"label={len(label_ohe.columns)} "
      f"key={len(key_ohe.columns)}")

# Galvenās dziesmu iezīmes, kas tiek izmantotas LIME izskaidrojumos
AUDIO_CORE = ["danceability","energy","loudness","tempo",
              "instrumentalness","popularity","duration_min",
              "engagement","release_year","explicit","mode"]


# 3.  Tiek simulēti lietotāju profili. Katrs lietotāja profils sastāv no mīļākā žanra, valsts, un dziesmu klausīšanās vēsturi
# 500 lietotāji
N_USERS = 500
np.random.seed(42)

user_genre   = np.random.choice(GENRES,    N_USERS)
user_country = np.random.choice(COUNTRIES, N_USERS)

user_rows, track_rows, play_rows = [], [], []

for uid in range(N_USERS):
    # Dziesmas, kas atbilst lietotāja žanram un valstij
    mask_both = ((df["genre"]   == user_genre[uid]) &
                 (df["country"] == user_country[uid])).values
    # Dziesmas, kas atbilst lietotāja žanram
    mask_genre = (df["genre"] == user_genre[uid]).values

    pool_both  = np.where(mask_both)[0]
    pool_genre = np.where(mask_genre)[0]

    n_both  = min(30, len(pool_both))
    n_genre = min(20, len(pool_genre))

    chosen_both  = np.random.choice(pool_both,  n_both,  replace=False) if n_both  > 0 else []
    chosen_genre = np.random.choice(pool_genre, n_genre, replace=False) if n_genre > 0 else []

    # Pievieno un nejaušību izvēlētas dziesmas
    random_tracks = np.random.choice(N_TRACKS, 10, replace=False)

    for tid in np.unique(np.concatenate([chosen_both, chosen_genre, random_tracks])):
        user_rows.append(uid)
        track_rows.append(tid)
    
        base_plays = int(df["engagement"].iloc[tid] * 50) + 1
        play_rows.append(np.random.randint(1, base_plays + 1))

plays_df = pd.DataFrame({
    "user_idx" : user_rows,
    "track_idx": track_rows,
    "plays"    : play_rows
}).drop_duplicates(["user_idx","track_idx"]).reset_index(drop=True)

R = csr_matrix(
    (plays_df["plays"].values,
     (plays_df["user_idx"].values, plays_df["track_idx"].values)),
    shape=(N_USERS, N_TRACKS)
)

print(f"\nUsers: {N_USERS}  |  Interactions: {len(plays_df):,}")


# 4.  Hibrīdā ieteikumu sistēma (SVD + kosinusa līdzība)

# Tiek izmantoti 40 vienumi un 40 lietotāji iezīmes, lai izveidotu matricu 
svd = TruncatedSVD(n_components=40, random_state=42)
user_factors = svd.fit_transform(R)       
item_factors = svd.components_.T          

print(f"SVD: lietotāju_iezīmes {user_factors.shape}  vienumu_iezīmes {item_factors.shape}")


def score_tracks(feature_batch: np.ndarray, user_idx: int) -> np.ndarray:

# Hibrīds vērtēšanas (scoring) funkcija, ko LIME izsauc katram perturbētajam paraugam
# feature_batch : (n_samples, n_features) — perturbēti dati
# Atgriež       : (n_samples,) atbilstības vērtības intervālā [0,1]

    user_vec = user_factors[user_idx]
    cf_all   = item_factors @ user_vec
    cf_norm  = (cf_all - cf_all.min()) / (cf_all.max() - cf_all.min() + 1e-9)

    listened = plays_df[plays_df["user_idx"] == user_idx]["track_idx"].values
    if len(listened):
        user_profile = feature_matrix[listened].mean(axis=0, keepdims=True)
    else:
        user_profile = feature_matrix.mean(axis=0, keepdims=True)

    scores = np.zeros(len(feature_batch))
    for i, fvec in enumerate(feature_batch):
        nearest   = np.argmin(np.linalg.norm(feature_matrix - fvec, axis=1))
        cf        = cf_norm[nearest]
        content   = float(cosine_similarity(fvec.reshape(1,-1), user_profile)[0, 0])
        # Ja dziesma ir populārāka vai to vairāk klausās, tad piešķir lielāku svaru
        pop_boost = fvec[FEATURE_NAMES.index("popularity")] * 0.05
        scores[i] = np.clip(0.55*cf + 0.40*content + pop_boost, 0, 1)

    return scores


def make_predictor(user_idx: int):
    def predictor(X):
        return score_tracks(np.asarray(X, dtype=float), user_idx)
    return predictor


# 5.  LIME implementācija

# Var izvēlēties lietotāju un dziesmu, kurai var noskaidrot, kāpēc šis vienums tiktu vai netiktu ieteikts
TARGET_USER  = 7
TARGET_TRACK = 1042  

meta  = df.iloc[TARGET_TRACK]

print(f"Izksaidro ieteikumu lietotājam {TARGET_USER}")
print(f"  Žanra / valsts preference (Genre / Country preference): {user_genre[TARGET_USER]} / {user_country[TARGET_USER]}")
print(f"  Dziesma (Track)  : {meta['track_name']}")
print(f"  Izpildītājs (Artist) : {meta['artist_name']}")
print(f"  Žanrs (Genre)  : {meta['genre']}  |  Country: {meta['country']}")
print(f"  Audio  : dance={meta['danceability']:.2f}  energy={meta['energy']:.2f}  "
      f"tempo={meta['tempo']:.0f}BPM  loudness={meta['loudness']:.1f}dB")
print(f"  Straumējumi (Streams): {meta['stream_count']:,}  |  Popularity: {meta['popularity']}")


background = shap_background = feature_matrix[
    np.random.choice(N_TRACKS, 2000, replace=False)
]

# LIME metodes izmantošana (tiek izmantota iebūvēta bibliotēka)
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data         = background,
    feature_names         = FEATURE_NAMES,
    mode                  = "regression",
    discretize_continuous = True,
    random_state          = 42
)

predictor   = make_predictor(TARGET_USER)
instance    = feature_matrix[TARGET_TRACK]

explanation = explainer.explain_instance(
    data_row    = instance,
    predict_fn  = predictor,
    num_features= 15,        # Tiek izvēlētas top-15 iezīmes, kas visvairāk ietekmē ieteikumu
    num_samples = 2000
)

print("\nLIME Top-15 Iezīmes:")

for feat, weight in sorted(explanation.as_list(), key=lambda x: abs(x[1]), reverse=True):
    bar  = "█" * max(1, int(abs(weight) * 80))
    sign = "+" if weight > 0 else "-"
    print(f"  {sign}{bar:<30s}  {feat:45s}  {weight:+.5f}")


# 6.  Tiek izmantoti vairāki "BATCH" katram žanram, lai izskaidrotu ieteikumu sistēmu globāli


genre_samples = {}
for g in sorted(GENRES):
    idxs = df[df["genre"] == g].index.tolist()
    genre_samples[g] = idxs[0]

print(f"\nBatch izskaidrojumu analīze {len(genre_samples)}")
batch_exps = {}
for g, tidx in genre_samples.items():
    exp = explainer.explain_instance(
        feature_matrix[tidx], predictor,
        num_features=len(AUDIO_CORE), num_samples=600
    )
    batch_exps[g] = dict(exp.as_list())


def _base_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    return fig, ax


def parse_to_core(lime_dict, core_features):
    # Pārveido datus atpakaļ to īstajos nosaukumos
    result = {f: 0.0 for f in core_features}
    for k, v in lime_dict.items():
        for fname in core_features:
            if fname in k:
                result[fname] += v
                break
    return result


# 8.  Vizualizācija (Waterfall diagramma)
 
def plot_waterfall(exp, track_idx, user_idx, save_path=None):
    items   = sorted(exp.as_list(), key=lambda x: x[1])
    labels  = [i[0] for i in items]
    weights = [i[1] for i in items]
    colors  = [GREEN if w >= 0 else RED for w in weights]
 
    fig, ax = _base_fig(13, 8)
    bars = ax.barh(labels, weights, color=colors, edgecolor="none", height=0.65)
    ax.axvline(0, color="#555", linewidth=1.2)
 
    for bar, w in zip(bars, weights):
        ax.text(w + (0.001 if w >= 0 else -0.001),
                bar.get_y() + bar.get_height()/2,
                f"{w:+.4f}", va="center",
                ha="left" if w >= 0 else "right",
                color="white", fontsize=8)
 
    m     = df.iloc[track_idx]
    score = predictor(feature_matrix[track_idx:track_idx+1])[0]
 
    ax.set_xlabel("LIME iezīmju ieguldījums", fontsize=11)
    ax.set_title(
        f'LIME izskaidrojums  ·  "{m["track_name"]}" — {m["artist_name"]}\n'
        f'Žanrs: {m["genre"]}  ·  Valsts: {m["country"]}  ·  '
        f'Prognozētais vērtējums lietotājam {user_idx}: {score:.3f}',
        fontsize=11, fontweight="bold", pad=14
    )
    pos_p = mpatches.Patch(color=GREEN, label="Palielina varbūtību, ka tiks ieteikts")
    neg_p = mpatches.Patch(color=RED,   label="Samazina varbūtību, ka tiks ieteikts")
    ax.legend(handles=[pos_p, neg_p], fontsize=9, loc="lower right")
    ax.grid(axis="x", linewidth=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()

plot_waterfall(explanation, TARGET_TRACK, TARGET_USER,
               save_path=os.path.join(OUT, "lime_spotify_waterfall.png"))