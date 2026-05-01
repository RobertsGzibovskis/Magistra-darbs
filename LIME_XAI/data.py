# data.py
# 1. Datu ielāde un apstrāde
# 2. Vienumu īpašību matrica
# 3. Lietotāju profilu simulācija

import warnings
warnings.filterwarnings("ignore")

import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix

import config


# Iezīmju definīcijas

CONTINUOUS = [
    "danceability", "energy", "loudness", "tempo",
    "instrumentalness", "popularity", "duration_min",
    "valence", "acousticness", "liveness", "speechiness", "year"
]

BINARY = ["explicit", "mode"]

# Pārvērš mākslinieku sarakstu (string vai list) par pirmo mākslinieka vārdu.
def _parse_artists(raw) -> str:
    try:
        lst = ast.literal_eval(raw)
        return lst[0] if lst else "Nezināms"
    except Exception:
        return str(raw)


# Ielādē un apstrādā Spotify datu kopu.

def load_tracks(csv_path: str) -> pd.DataFrame:

    print("Datukopa")
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

    # Apstrādāt tikai pirmos N ierakstus
    if config.TRACK_SAMPLE_N is not None:
        df = df.sample(n=min(config.TRACK_SAMPLE_N, len(df)),
                       random_state=config.RANDOM_SEED).reset_index(drop=True)
        print(f"  [Paraugs: {len(df):,} no visiem ierakstiem]")

    print(f"  Rindas: {len(df):,}  |  Kolonnas: {df.columns.tolist()}")

    # Primārais mākslinieks
    df["artist_name"] = df["artists"].apply(_parse_artists)

    # Ilgums minūtēs
    df["duration_min"] = df["duration_ms"] / 60_000

    # Dekāde — izmantojam kā vieglu žanra aizstājēju lietotāja profilā
    df["decade"] = (df["year"] // 10) * 10

    print(f"  Gadi    : {int(df['year'].min())} – {int(df['year'].max())}")
    print(f"  Dekādes : {sorted(df['decade'].unique().tolist())}")
    return df

# Izveido vienumu īpašību matricu

def build_feature_matrix(df: pd.DataFrame):

    key_ohe    = pd.get_dummies(df["key"],    prefix="key")
    decade_ohe = pd.get_dummies(df["decade"], prefix="dec")

    raw = pd.concat([
        df[CONTINUOUS].reset_index(drop=True),
        df[BINARY].reset_index(drop=True),
        key_ohe.reset_index(drop=True),
        decade_ohe.reset_index(drop=True),
    ], axis=1).astype(float)

    feature_names  = raw.columns.tolist()
    scaler         = MinMaxScaler()
    feature_matrix = scaler.fit_transform(raw)

    print(f"\nVienumu Matrica: {feature_matrix.shape}")
    print(f"  Continuous : {CONTINUOUS}")
    print(f"  Binary     : {BINARY}")
    print(f"  OHE dims   : key={len(key_ohe.columns)}  decade={len(decade_ohe.columns)}")

    return feature_matrix, feature_names, scaler


#  Simulē lietotāju klausīšanās vēsturi. Katram lietotājam tiek piešķirta mīļākā dekāde un enerģijas preference.

def simulate_users(df: pd.DataFrame, feature_matrix: np.ndarray,
                   n_users: int = config.N_USERS,
                   seed: int = config.RANDOM_SEED):


    np.random.seed(seed)
    decades  = df["decade"].unique().tolist()
    n_tracks = len(df)

    # Katram lietotājam
    user_decade  = np.random.choice(decades, n_users)
    user_energy  = np.random.choice(["high", "low"], n_users)

    user_rows, track_rows, play_rows = [], [], []

    for uid in range(n_users):
        mask_decade = (df["decade"] == user_decade[uid]).values

        if user_energy[uid] == "high":
            mask_energy = (df["energy"] >= 0.6).values
        else:
            mask_energy = (df["energy"] < 0.6).values

        pool_both   = np.where(mask_decade & mask_energy)[0]
        pool_decade = np.where(mask_decade)[0]

        n_both   = min(30, len(pool_both))
        n_decade = min(20, len(pool_decade))

        chosen_both   = (np.random.choice(pool_both,   n_both,   replace=False)
                         if n_both   > 0 else [])
        chosen_decade = (np.random.choice(pool_decade, n_decade, replace=False)
                         if n_decade > 0 else [])

        random_tracks = np.random.choice(n_tracks, 10, replace=False)

        for tid in np.unique(np.concatenate([chosen_both, chosen_decade,
                                              random_tracks])):
            user_rows.append(uid)
            track_rows.append(int(tid))
            pop_norm  = float(df["popularity"].iloc[int(tid)]) / 100.0
            base_plays = max(1, int(pop_norm * 50))
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
    return plays_df, R, user_decade, user_energy
