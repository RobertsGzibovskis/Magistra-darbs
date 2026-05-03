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

 
PREF_VALUES = {
    "decade"      : ["1920","1930","1940","1950","1960",
                     "1970","1980","1990","2000","2010","2020"],
    "energy"      : ["high", "low"],
    "danceability": ["high", "low"],
    "valence"     : ["happy", "sad"],
    "popularity"  : ["mainstream", "niche"],
}
 
PREF_LABELS = {
    "decade"      : "Dekāde",
    "energy"      : "Enerģija",
    "danceability": "Dejojamība",
    "valence"     : "Noskaņojums",
    "popularity"  : "Popularitāte",
}



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

 # Preferneču maska
def _preference_mask(df: pd.DataFrame, pref: str, value: str) -> np.ndarray:
    
    if pref == "decade":
        return (df["decade"] == int(value)).values
    if pref == "energy":
        return (df["energy"] >= 0.60).values if value == "high" \
               else (df["energy"] < 0.60).values
    if pref == "danceability":
        return (df["danceability"] >= 0.60).values if value == "high" \
               else (df["danceability"] < 0.60).values
    if pref == "valence":
        return (df["valence"] >= 0.55).values if value == "happy" \
               else (df["valence"] < 0.55).values
    if pref == "popularity":
        return (df["popularity"] >= 40).values if value == "mainstream" \
               else (df["popularity"] < 40).values
    return np.ones(len(df), dtype=bool)


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


#  Simulē lietotāju klausīšanās vēsturi. Katram lietotājam tiek piešķirta mīļākā dekāde, enerģijas preference, dejojamība, noskaņojums, popularitāte.

def simulate_users(df: pd.DataFrame,
                   feature_matrix: np.ndarray,
                   n_users: int = config.N_USERS,
                   seed: int    = config.RANDOM_SEED):
 
    np.random.seed(seed)
    n_tracks = len(df)
 
    # Katram lietotājam — nejaušas preferences
    user_prefs = {
        pref: np.random.choice(vals, n_users)
        for pref, vals in PREF_VALUES.items()
    }
 
    # Priekšaprēķins: maskas visām preferences vērtībām
    masks = {
        pref: {v: _preference_mask(df, pref, v) for v in vals}
        for pref, vals in PREF_VALUES.items()
    }
 
    user_rows, track_rows, play_rows = [], [], []
 
    for uid in range(n_users):
 
        # Šī lietotāja maskas
        uid_masks = [
            masks[pref][user_prefs[pref][uid]]
            for pref in PREF_VALUES
        ]
 
        # Kārta A: visas 5 preferences
        mask_all5 = np.ones(n_tracks, dtype=bool)
        for m in uid_masks:
            mask_all5 &= m
 
        # Kārta B: vismaz 3 no 5
        pref_sum  = sum(m.astype(int) for m in uid_masks)
        mask_3of5 = (pref_sum >= 3) & ~mask_all5
 
        pool_all5 = np.where(mask_all5)[0]
        pool_3of5 = np.where(mask_3of5)[0]
 
        n_all5 = min(30, len(pool_all5))
        n_3of5 = min(25, len(pool_3of5))
 
        chosen_all5 = (np.random.choice(pool_all5, n_all5, replace=False)
                       if n_all5 > 0 else np.array([], dtype=int))
        chosen_3of5 = (np.random.choice(pool_3of5, n_3of5, replace=False)
                       if n_3of5 > 0 else np.array([], dtype=int))
 
        random_tracks = np.random.choice(n_tracks, 10, replace=False)
 
        chosen = np.unique(np.concatenate([chosen_all5, chosen_3of5, random_tracks]))
 
        for tid in chosen:
            user_rows.append(uid)
            track_rows.append(int(tid))
            pop_norm   = float(df["popularity"].iloc[int(tid)]) / 100.0
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
    print(f"  Vidēji dziesmas uz lietotāju : {len(plays_df)/n_users:.1f}")
    print()
    print("  Lietotāja 0 preferences (piemērs):")
    for pref, label in PREF_LABELS.items():
        print(f"    {label:15s}: {user_prefs[pref][0]}")
 
    return plays_df, R, user_prefs