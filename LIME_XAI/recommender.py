# recommender.py
# Hibrīdā ieteikumu sistēma un LIME explainer inicializācija


import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

import lime
import lime.lime_tabular

import config

# Apmāca SVD modeli uz lietotājs × dziesma spēlēšanas matricas.
def build_svd_model(R: csr_matrix,
                    n_components: int = config.SVD_COMPONENTS,
                    seed: int = config.RANDOM_SEED):
  
    svd          = TruncatedSVD(n_components=n_components, random_state=seed)
    user_factors = svd.fit_transform(R)
    item_factors = svd.components_.T

    print(f"SVD: lietotāju_iezīmes {user_factors.shape}  "
          f"vienumu_iezīmes {item_factors.shape}")
    return user_factors, item_factors

# Izveido hibrīdo scoring funkciju konkrētam lietotājam. Hibrīds: 0.55 × SVD (CF) + 0.40 × kosinusa līdzība + 0.05 × popularitāte
def make_scorer(user_factors: np.ndarray,
                item_factors: np.ndarray,
                feature_matrix: np.ndarray,
                feature_names: list,
                plays_df):
 
    pop_idx = feature_names.index("popularity")

# Hibrīdā vērtēšanas funkcija, ko LIME izsauc katram perturbētajam paraugam.

    def score_tracks(feature_batch: np.ndarray, user_idx: int) -> np.ndarray:
      
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
            content   = float(cosine_similarity(
                            fvec.reshape(1, -1), user_profile)[0, 0])
            # Populārākām dziesmām piešķir nelielu papildu svaru
            pop_boost = fvec[pop_idx] * 0.05
            scores[i] = np.clip(0.55 * cf + 0.40 * content + pop_boost, 0, 1)

        return scores

    def make_predictor(user_idx: int):
     
        def predictor(X):
            return score_tracks(np.asarray(X, dtype=float), user_idx)
        return predictor

    return make_predictor

# Inicializē LimeTabularExplainer.
def build_lime_explainer(feature_matrix: np.ndarray,
                         feature_names: list,
                         n_background: int = config.LIME_BACKGROUND_N,
                         seed: int = config.RANDOM_SEED):
 
    np.random.seed(seed)
    background = feature_matrix[
        np.random.choice(len(feature_matrix), n_background, replace=False)
    ]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data         = background,
        feature_names         = feature_names,
        mode                  = "regression",
        discretize_continuous = True,
        random_state          = seed,
    )
    return explainer, background
