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

# Galvenās audio iezīmes — tieši interpretējamas
AUDIO_FEATURES = [
    "valence",          
    "danceability",     
    "energy",           
    "acousticness",     
    "instrumentalness", 
    "liveness",         
    "speechiness",     
    "loudness",         
    "tempo",            
    "popularity",       
    "year",             
    "duration_min",     
]

# Era-deviation iezīmes 
ERA_DEV_FEATURES = [
    "energy_vs_era",   
    "valence_vs_era",   
    "dance_vs_era",     
    "acoustic_vs_era",  
]

# Bināras iezīmes
BINARY_FEATURES = [
    "explicit",        
    "mode",            
]

# Populārākie žanri, ko izmantojam kā OHE iezīmes
TOP_GENRES = [
    "rock", "pop", "hip hop", "rap", "dance pop", "country",
    "jazz", "classical", "r&b", "metal", "indie", "electronic",
    "folk", "latin", "soul",
]

# Vērtības un etiketes lietotāju preferenču simulācijai
PREF_VALUES = {
    "energy"      : ["high", "low"],
    "danceability": ["high", "low"],
    "valence"     : ["happy", "sad"],
    "popularity"  : ["mainstream", "niche"],
    "era"         : ["classic", "modern"],  # classic=pirms 1990, modern=pēc 1990
}

PREF_LABELS = {
    "energy"      : "Enerģija",
    "danceability": "Dejojamība",
    "valence"     : "Noskaņojums",
    "popularity"  : "Popularitāte",
    "era"         : "Laikmets",
}


# ── Palīgfunkcijas ────────────────────────────────────────────────────────────

def _parse_first_artist(raw) -> str:
    try:
        lst = ast.literal_eval(str(raw))
        return lst[0].strip().lower() if lst else ""
    except Exception:
        return str(raw).strip().lower()


def _parse_genre_list(raw) -> list:
    try:
        lst = ast.literal_eval(str(raw))
        return [g.strip().lower() for g in lst] if isinstance(lst, list) else []
    except Exception:
        return []

# Preferneču maska
def _preference_mask(df: pd.DataFrame, pref: str, value: str) -> np.ndarray:
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
    if pref == "era":
        return (df["year"] < 1990).values if value == "classic" \
               else (df["year"] >= 1990).values
    return np.ones(len(df), dtype=bool)


# Ielādē un apstrādā Spotify datu kopu.

def load_tracks(csv_main: str, csv_genres: str, csv_by_year: str) -> pd.DataFrame:
  
    print("Datukopa: Spotify (data + genres + by_year)")

    # 1. Galvenā datu kopa
    df = pd.read_csv(csv_main).dropna().reset_index(drop=True)

    if config.TRACK_SAMPLE_N is not None:
        df = df.sample(
            n=min(config.TRACK_SAMPLE_N, len(df)),
            random_state=config.RANDOM_SEED
        ).reset_index(drop=True)
        print(f"  [Paraugs: {len(df):,} dziesmas]")

    df["artist_name"]  = df["artists"].apply(_parse_first_artist)
    df["duration_min"] = df["duration_ms"] / 60_000

    # 2. Era-deviation iezīmes no data_by_year.csv
    by_year = pd.read_csv(csv_by_year)[
        ["year", "energy", "valence", "danceability", "acousticness"]
    ].rename(columns={
        "energy"      : "era_energy",
        "valence"     : "era_valence",
        "danceability": "era_dance",
        "acousticness": "era_acoustic",
    })

    df = df.merge(by_year, on="year", how="left")
    df["energy_vs_era"]   = df["energy"]       - df["era_energy"].fillna(df["energy"].mean())
    df["valence_vs_era"]  = df["valence"]       - df["era_valence"].fillna(df["valence"].mean())
    df["dance_vs_era"]    = df["danceability"]  - df["era_dance"].fillna(df["danceability"].mean())
    df["acoustic_vs_era"] = df["acousticness"]  - df["era_acoustic"].fillna(df["acousticness"].mean())

    # 3. Žanri no data_w_genres.csv
    wg = pd.read_csv(csv_genres)
    wg["_artist_key"] = wg["artists"].apply(
        lambda x: str(x).strip().strip('"').lower()
    )
    wg["genres_list"] = wg["genres"].apply(_parse_genre_list)

    genre_lookup = dict(zip(wg["_artist_key"], wg["genres_list"]))
    df["genres_list"] = df["artist_name"].map(genre_lookup).apply(
        lambda x: x if isinstance(x, list) else []
    )

    # OHE — vai kāds no TOP_GENRES parādās dziesmas mākslinieka žanru sarakstā
    for genre in TOP_GENRES:
        col = f"genre_{genre.replace(' ', '_')}"
        df[col] = df["genres_list"].apply(
            lambda gl: int(any(genre in g for g in gl))
        )

    genre_coverage = (df["genres_list"].apply(len) > 0).mean()
    print(f"  Žanru pārklājums: {genre_coverage:.1%}")
    print(f"  Gadi: {int(df['year'].min())} – {int(df['year'].max())}")
    print(f"  Rindas: {len(df):,}")

    return df

# Izveido vienumu īpašību matricu
def build_feature_matrix(df: pd.DataFrame):

    genre_cols = [f"genre_{g.replace(' ', '_')}" for g in TOP_GENRES
                  if f"genre_{g.replace(' ', '_')}" in df.columns]

    all_cols = AUDIO_FEATURES + ERA_DEV_FEATURES + BINARY_FEATURES + genre_cols

    raw           = df[all_cols].fillna(0).astype(float)
    feature_names = raw.columns.tolist()
    scaler        = MinMaxScaler()
    feature_matrix = scaler.fit_transform(raw)

    print(f"\nIezīmju matrica: {feature_matrix.shape}")
    print(f"  Audio     ({len(AUDIO_FEATURES)}): {AUDIO_FEATURES}")
    print(f"  Era-dev   ({len(ERA_DEV_FEATURES)}): {ERA_DEV_FEATURES}")
    print(f"  Bināras   ({len(BINARY_FEATURES)}): {BINARY_FEATURES}")
    print(f"  Žanri OHE ({len(genre_cols)}): {[g.replace('genre_','') for g in genre_cols]}")

    return feature_matrix, feature_names, scaler


#  Simulē lietotāju klausīšanās vēsturi. Katram lietotājam tiek piešķirta mīļākā dekāde, enerģijas preference, dejojamība, noskaņojums, popularitāte.
def simulate_users(df: pd.DataFrame,
                   feature_matrix: np.ndarray,
                   n_users: int = config.N_USERS,
                   seed: int    = config.RANDOM_SEED):
    np.random.seed(seed)
    n_tracks = len(df)

    user_prefs = {
        pref: np.random.choice(vals, n_users)
        for pref, vals in PREF_VALUES.items()
    }

    masks = {
        pref: {v: _preference_mask(df, pref, v) for v in vals}
        for pref, vals in PREF_VALUES.items()
    }

    user_rows, track_rows, play_rows = [], [], []

    for uid in range(n_users):
        uid_masks = [masks[p][user_prefs[p][uid]] for p in PREF_VALUES]

        mask_all = np.ones(n_tracks, dtype=bool)
        for m in uid_masks:
            mask_all &= m

        pref_sum   = sum(m.astype(int) for m in uid_masks)
        mask_3of5  = (pref_sum >= 3) & ~mask_all

        pool_all  = np.where(mask_all)[0]
        pool_3of5 = np.where(mask_3of5)[0]

        chosen_all  = (np.random.choice(pool_all,  min(30, len(pool_all)),  replace=False)
                       if len(pool_all)  > 0 else np.array([], dtype=int))
        chosen_3of5 = (np.random.choice(pool_3of5, min(25, len(pool_3of5)), replace=False)
                       if len(pool_3of5) > 0 else np.array([], dtype=int))
        random_tracks = np.random.choice(n_tracks, 10, replace=False)

        for tid in np.unique(np.concatenate([chosen_all, chosen_3of5, random_tracks])):
            user_rows.append(uid)
            track_rows.append(int(tid))
            pop_norm  = float(df["popularity"].iloc[int(tid)]) / 100.0
            play_rows.append(np.random.randint(1, max(2, int(pop_norm * 50) + 1)))

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
    print(f"  Vidēji dziesmas uz lietotāju: {len(plays_df)/n_users:.1f}")
    print("\n  Lietotāja 0 preferences (piemērs):")
    for pref, label in PREF_LABELS.items():
        print(f"    {label:15s}: {user_prefs[pref][0]}")

    return plays_df, R, user_prefs