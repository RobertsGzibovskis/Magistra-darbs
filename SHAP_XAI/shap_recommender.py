# shap_recommender.py
# Hibrīdā ieteikumu sistēma + SHAP SamplingExplainer

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

import shap

import shap_config as config

_KNN_K = 5

# 14 interpretējamas audio/metadatu iezīmes — SHAP izmanto šīs
CORE_FEATURES = [
    "danceability", "energy", "loudness", "tempo",
    "instrumentalness", "popularity", "duration_min",
    "valence", "acousticness", "liveness", "speechiness",
    "year", "explicit", "mode"
]


def build_svd_model(R: csr_matrix,
                    n_components: int = config.SVD_COMPONENTS,
                    seed: int = config.RANDOM_SEED):
    svd          = TruncatedSVD(n_components=n_components, random_state=seed)
    user_factors = svd.fit_transform(R)
    item_factors = svd.components_.T
    print(f"SVD: lietotāju_iezīmes {user_factors.shape}  "
          f"vienumu_iezīmes {item_factors.shape}")
    return user_factors, item_factors

 
# Izveido hibrīdo scoring funkciju.
# Hibrīds: 0.55 × CF (k-NN interpolēts) + 0.40 × kosinusa līdzība + 0.05 × popularitāte
def make_scorer(user_factors, item_factors, feature_matrix, feature_names, plays_df):
    pop_idx = feature_names.index("popularity")

    def _cf_scores(user_idx):
        uv = user_factors[user_idx]
        cf = item_factors @ uv
        return (cf - cf.min()) / (cf.max() - cf.min() + 1e-9)

    def score_tracks(feature_batch, user_idx):
        cf_all = _cf_scores(user_idx)
        listened = plays_df[plays_df["user_idx"] == user_idx]["track_idx"].values
        user_profile = (feature_matrix[listened].mean(axis=0, keepdims=True)
                        if len(listened) else feature_matrix.mean(axis=0, keepdims=True))
        scores = np.zeros(len(feature_batch))
        for i, fvec in enumerate(feature_batch):
            dists   = np.linalg.norm(feature_matrix - fvec, axis=1)
            knn_idx = np.argpartition(dists, _KNN_K)[:_KNN_K]
            weights = 1.0 / (dists[knn_idx] + 1e-6)
            cf      = float(np.average(cf_all[knn_idx], weights=weights))
            content = float(cosine_similarity(fvec.reshape(1,-1), user_profile)[0,0])
            scores[i] = np.clip(0.55*cf + 0.40*content + fvec[pop_idx]*0.05, 0, 1)
        return scores

    def make_predictor(user_idx):
        def predictor(X):
            return score_tracks(np.asarray(X, dtype=float), user_idx)
        return predictor

    return make_predictor

# Inicializē  Izveido SHAP SamplingExplainer
def build_shap_explainer(feature_matrix, feature_names, predictor,
                          n_background=config.SHAP_BACKGROUND_N,
                          seed=config.RANDOM_SEED):

    np.random.seed(seed)

    # Iegūst svarīgo kolonnu indeksus
    core_idx = [feature_names.index(f) for f in CORE_FEATURES]
    fm_core  = feature_matrix[:, core_idx]
    fm_mean  = feature_matrix.mean(axis=0)   # fona vektors kolonnām

    # Wrappers: SHAP dod svarīgo iezīmju vektoru
    def predictor_core(X_core):
        X_full = np.tile(fm_mean, (len(X_core), 1))
        X_full[:, core_idx] = np.asarray(X_core, dtype=float)
        return predictor(X_full)

    background = shap.sample(fm_core, n_background, random_state=seed)
    expected_value = float(np.mean(predictor_core(background)))

    explainer = shap.SamplingExplainer(
        model = predictor_core,
        data  = background,
    )

    print(f"SHAP SamplingExplainer: fons={n_background}  "
          f"core_iezīmes={len(CORE_FEATURES)}  E[f]={expected_value:.5f}")

    return explainer, background, core_idx, fm_core, expected_value, predictor_core
