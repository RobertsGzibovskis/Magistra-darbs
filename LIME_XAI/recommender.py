# recommender.py
# Hibrīdā ieteikumu sistēma un LIME explainer inicializācija

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

import lime
import lime.lime_tabular

import config

_KNN_K = 5  # kaimiņu skaits CF interpolācijai

def build_svd_model(R: csr_matrix,
                    n_components: int = config.SVD_COMPONENTS,
                    seed: int = config.RANDOM_SEED):
    svd          = TruncatedSVD(n_components=n_components, random_state=seed)
    user_factors = svd.fit_transform(R)
    item_factors = svd.components_.T

    print(f"SVD: lietotāju_iezīmes {user_factors.shape}  "
          f"vienumu_iezīmes {item_factors.shape}")
    return user_factors, item_factors


def make_scorer(user_factors: np.ndarray,
                item_factors: np.ndarray,
                feature_matrix: np.ndarray,
                feature_names: list,
                plays_df):

    pop_idx = feature_names.index("popularity")
    n_tracks = len(feature_matrix)

    # Sagatavojam Collaborative Filtering (CF) prognozes visām dziesmām iepriekš
    cf_all_users = np.zeros((len(user_factors), n_tracks))
    for uid in range(len(user_factors)):
        u_vec = user_factors[uid]
        raw_cf = item_factors @ u_vec
        cf_all_users[uid] = (raw_cf - raw_cf.min()) / (raw_cf.max() - raw_cf.min() + 1e-9)

    def score_tracks(feature_batch: np.ndarray, user_idx: int, base_cf_score: float = None):
        user_profile = np.zeros(len(feature_names))
        user_history = plays_df[plays_df["user_idx"] == user_idx]["track_idx"].values
        
        if len(user_history) > 0:
            user_profile = np.mean(feature_matrix[user_history], axis=0)
        else:
            user_profile = np.mean(feature_matrix, axis=0)

        cf_all = cf_all_users[user_idx]
        scores = np.zeros(len(feature_batch))
        
        for i, fvec in enumerate(feature_batch):
            # Tā kā LIME tagad sūta perturbētus vektorus BEZ žanriem (īsākus),
            # mums tie ir jāpapildina līdz pilnajam izmēram ar target dziesmas žanriem fona aprēķiniem.
            if len(fvec) < len(feature_names):
                full_fvec = np.zeros(len(feature_names))
                full_fvec[:len(fvec)] = fvec
                # Atlikušo daļu (žanrus) paņemam no target dziesmas (tie netiek perturbēti)
                full_fvec[len(fvec):] = feature_matrix[config.TARGET_TRACK, len(fvec):]
            else:
                full_fvec = fvec

            # Ja ir iedots fiksēts CF, izmantojam to (novērš k-NN sabrukšanu perturbāciju laikā)
            if base_cf_score is not None:
                cf = base_cf_score
            else:
                dists   = np.linalg.norm(feature_matrix - full_fvec, axis=1)
                knn_idx = np.argpartition(dists, _KNN_K)[:_KNN_K]
                weights = 1.0 / (dists[knn_idx] + 1e-6)
                cf      = float(np.average(cf_all[knn_idx], weights=weights))

            content   = float(cosine_similarity(full_fvec.reshape(1, -1), user_profile.reshape(1, -1))[0, 0])
            pop_boost = full_fvec[pop_idx] * 0.05
            scores[i] = np.clip(0.60 * cf + 0.40 * content + pop_boost, 0, 1)

        return scores

    def make_predictor(user_idx: int, base_cf_score: float = None):
        def predictor(X):
            return score_tracks(np.asarray(X, dtype=float), user_idx, base_cf_score=base_cf_score)
        return predictor

    return make_predictor


def build_lime_explainer(feature_matrix: np.ndarray,
                         feature_names: list,
                         n_background: int = config.LIME_BACKGROUND_N,
                         seed: int = config.RANDOM_SEED):
    
    # Atrodam robežu, kur beidzas audio/laikmeta/binārās iezīmes un sākas žanri
    non_genre_indices = [i for i, name in enumerate(feature_names) if not name.startswith("genre_")]
    
    # Filtrējam fona datus — padodam LIME tikai iezīmes BEZ žanriem
    filtered_matrix = feature_matrix[:, non_genre_indices]
    filtered_names = [feature_names[i] for i in non_genre_indices]

    np.random.seed(seed)
    background = filtered_matrix[
        np.random.choice(len(filtered_matrix), n_background, replace=False)
    ]

    # Atrodam bināro iezīmju indeksus jaunajā filtrētajā sarakstā
    categorical_features = []
    if "explicit" in filtered_names:
        categorical_features.append(filtered_names.index("explicit"))
    if "mode" in filtered_names:
        categorical_features.append(filtered_names.index("mode"))

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=background,
        feature_names=filtered_names,
        class_names=["Score"],
        categorical_features=categorical_features,
        mode="regression",
        kernel_width=3.0,
        verbose=False,
        random_state=seed
    )
    
    return explainer, filtered_names