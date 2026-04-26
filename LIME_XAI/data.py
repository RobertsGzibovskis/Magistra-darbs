# data.py
# 1. Datu ielāde un apstrāde
# 2. Vienumu īpašību matrica (audio + metadati + OHE)
# 3. Lietotāju profilu simulācija

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix

import config


# Iezīmju definīcijas

CONTINUOUS = [
    "danceability", "energy", "loudness", "tempo",
    "instrumentalness", "popularity", "duration_min",
    "engagement", "release_year"
]

BINARY = ["explicit", "mode"]

# Galvenās audio iezīmes LIME izskaidrojumiem (bez OHE kolonnām)
AUDIO_CORE = [
    "danceability", "energy", "loudness", "tempo",
    "instrumentalness", "popularity", "duration_min",
    "engagement", "release_year", "explicit", "mode"
]

#    Ielādē un apstrādā Spotify datu kopu. Pievieno: release_year, engagement (normalizēts), duration_min.
def load_tracks(csv_path: str) -> pd.DataFrame:
 
    print("Datukopa")
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    print(f"  Rindas: {len(df):,}  |  Kolonnas: {df.columns.tolist()}")

    # Dziesmas izlaišanas gads
    df["release_year"] = pd.to_datetime(df["release_date"]).dt.year

    # Straumējumu skaits
    df["engagement"] = np.log1p(df["stream_count"])
    df["engagement"] = (
        (df["engagement"] - df["engagement"].min()) /
        (df["engagement"].max() - df["engagement"].min())
    )

    # Ilgums minūtēs (pārveidots no ms)
    df["duration_min"] = df["duration_ms"] / 60_000

    print(f"  Žanri  : {sorted(df['genre'].unique().tolist())}")
    print(f"  Valstis: {df['country'].nunique()}")
    return df

#  Izveido vienumu īpašību matricu:
#  skaitliskas iezīmes (CONTINUOUS + BINARY)
#  OHE: žanrs, valsts, izdevniecība, tonalitāte

def build_feature_matrix(df: pd.DataFrame):
  
    genre_ohe   = pd.get_dummies(df["genre"],   prefix="g")
    country_ohe = pd.get_dummies(df["country"], prefix="c")
    label_ohe   = pd.get_dummies(df["label"],   prefix="lbl")
    key_ohe     = pd.get_dummies(df["key"],     prefix="key")

    raw = pd.concat([
        df[CONTINUOUS].reset_index(drop=True),
        df[BINARY].reset_index(drop=True),
        genre_ohe.reset_index(drop=True),
        country_ohe.reset_index(drop=True),
        label_ohe.reset_index(drop=True),
        key_ohe.reset_index(drop=True),
    ], axis=1).astype(float)

    feature_names  = raw.columns.tolist()
    scaler         = MinMaxScaler()
    feature_matrix = scaler.fit_transform(raw)

    print(f"\nVienumu Matrica: {feature_matrix.shape}")
    print(f"  Continuous : {CONTINUOUS}")
    print(f"  Binary     : {BINARY}")
    print(f"  OHE dims   : genre={len(genre_ohe.columns)} "
          f"country={len(country_ohe.columns)} "
          f"label={len(label_ohe.columns)} "
          f"key={len(key_ohe.columns)}")

    return feature_matrix, feature_names, scaler

# Simulē lietotāju klausīšanās vēsturi.
# Katram lietotājam tiek piešķirts mīļākais žanrs un valsts.
# Dziesmas tiek izvēlētas pēc žanra/valsts atbilstības + nejaušības.
def simulate_users(df: pd.DataFrame, feature_matrix: np.ndarray,
                   n_users: int = config.N_USERS,
                   seed: int = config.RANDOM_SEED):

    np.random.seed(seed)
    genres    = df["genre"].unique().tolist()
    countries = df["country"].unique().tolist()
    n_tracks  = len(df)

    user_genre   = np.random.choice(genres,    n_users)
    user_country = np.random.choice(countries, n_users)

    user_rows, track_rows, play_rows = [], [], []

    for uid in range(n_users):
        mask_both  = ((df["genre"]   == user_genre[uid]) &
                      (df["country"] == user_country[uid])).values
        mask_genre = (df["genre"] == user_genre[uid]).values

        pool_both  = np.where(mask_both)[0]
        pool_genre = np.where(mask_genre)[0]

        n_both  = min(30, len(pool_both))
        n_genre = min(20, len(pool_genre))

        chosen_both  = (np.random.choice(pool_both,  n_both,  replace=False)
                        if n_both  > 0 else [])
        chosen_genre = (np.random.choice(pool_genre, n_genre, replace=False)
                        if n_genre > 0 else [])

        # Pievieno dziesmas pēc nejaušības
        random_tracks = np.random.choice(n_tracks, 10, replace=False)

        for tid in np.unique(np.concatenate([chosen_both, chosen_genre,
                                              random_tracks])):
            user_rows.append(uid)
            track_rows.append(int(tid))
            base_plays = int(df["engagement"].iloc[int(tid)] * 50) + 1
            play_rows.append(np.random.randint(1, base_plays + 1))

    plays_df = pd.DataFrame({
        "user_idx":  user_rows,
        "track_idx": track_rows,
        "plays":     play_rows,
    }).drop_duplicates(["user_idx", "track_idx"]).reset_index(drop=True)

    R = csr_matrix(
        (plays_df["plays"].values,
         (plays_df["user_idx"].values, plays_df["track_idx"].values)),
        shape=(n_users, n_tracks)
    )

    print(f"\nLietotāji: {n_users}  |  Mijiedarbības: {len(plays_df):,}")
    return plays_df, R, user_genre, user_country
