# shap_main.py — SHAP SamplingExplainer versija (lokāls skaidrojums)

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np

import shap_config as config
sys.modules["config"] = config

import shap_data as data
import shap_recommender as recommender
import shap_evaluation  as evaluation
from shap_recommender import CORE_FEATURES

# Izveido ieteikumu, balstoties uz iezīmēm
def find_top_recommendation(user_idx, plays_df, user_factors, item_factors):
    already_heard = set(plays_df[plays_df["user_idx"]==user_idx]["track_idx"].values)
    uv = user_factors[user_idx]
    cf = item_factors @ uv
    cf_norm = (cf - cf.min()) / (cf.max() - cf.min() + 1e-9)
    mask = np.ones(len(cf_norm), dtype=bool)
    for idx in already_heard:
        mask[idx] = False
    cf_norm[~mask] = -1.0
    return int(np.argmax(cf_norm)), float(np.max(cf_norm))


def main():
  
    print("SHAP XAI")

    # 1. Dati un modelis
    df = data.load_tracks(config.CSV)
    feature_matrix, feature_names, scaler = data.build_feature_matrix(df)
    plays_df, R, user_prefs = data.simulate_users(
        df, feature_matrix, n_users=config.N_USERS, seed=config.RANDOM_SEED)
    user_factors, item_factors = recommender.build_svd_model(
        R, n_components=config.SVD_COMPONENTS, seed=config.RANDOM_SEED)
    make_predictor = recommender.make_scorer(
        user_factors, item_factors, feature_matrix, feature_names, plays_df)

    TARGET_USER = config.TARGET_USER
    predictor   = make_predictor(TARGET_USER)

    # 2. SHAP SamplingExplainer (pamata iezīmes)
    print(f"\nVeido SHAP SamplingExplainer lietotājam {TARGET_USER} ...")
    explainer, background, core_idx, fm_core, expected_value, predictor_core = \
        recommender.build_shap_explainer(
            feature_matrix, feature_names, predictor,
            n_background=config.SHAP_BACKGROUND_N,
            seed=config.RANDOM_SEED)

    # 3. Ieteikuma izvēle
    if config.TARGET_TRACK == -1:
        print(f"\nMeklē labāko ieteikumu lietotājam {TARGET_USER} ...")
        TARGET_TRACK, rec_score = find_top_recommendation(
            TARGET_USER, plays_df, user_factors, item_factors)
        print(f"Ieteikts: track_idx={TARGET_TRACK}  score={rec_score:.4f}")
    else:
        TARGET_TRACK = config.TARGET_TRACK
        rec_score = float(predictor(feature_matrix[[TARGET_TRACK]])[0])
        print(f"Izmanto TARGET_TRACK={TARGET_TRACK}  score={rec_score:.4f}")

    meta          = df.iloc[TARGET_TRACK]
    instance_full = feature_matrix[TARGET_TRACK]
    instance_core = fm_core[TARGET_TRACK]

    print(f"\nIeteikums lietotājam {TARGET_USER}")
    print(f"  Preferences :")
    for pref, label in data.PREF_LABELS.items():
        print(f"    {label:15s}: {user_prefs[pref][TARGET_USER]}")
    print(f"  Dziesma     : {meta['name']}")
    print(f"  Izpildītājs : {meta['artist_name']}")
    print(f"  Gads/Dekāde : {int(meta['year'])} / {int(meta['decade'])}")
    print(f"  Audio       : dance={meta['danceability']:.2f}  energy={meta['energy']:.2f}  "
          f"valence={meta['valence']:.2f}  tempo={meta['tempo']:.0f}BPM")
    print(f"  Popularitāte: {meta['popularity']}")

    # 4. SHAP vērtību aprēķins (2 reizes)
    print(f"\nAprēķina SHAP vērtības (nsamples={config.SHAP_LOCAL_NSAMPLES}) ...")
    np.random.seed(config.RANDOM_SEED)
    sv1 = np.asarray(explainer.shap_values(
        instance_core.reshape(1,-1),
        nsamples=config.SHAP_LOCAL_NSAMPLES, silent=True)).flatten()

    np.random.seed(config.RANDOM_SEED + 1)
    sv2 = np.asarray(explainer.shap_values(
        instance_core.reshape(1,-1),
        nsamples=config.SHAP_LOCAL_NSAMPLES, silent=True)).flatten()

    # 5. Top iezīmes
    print("SHAP LOKĀLAIS SKAIDROJUMS")
    print(f"\n  Bāzlīnija E[f]     = {expected_value:.5f}")
    print(f"  Σ phi_i            = {sv1.sum():.5f}")
    print(f"  Prognoze (E+Σphi)  = {expected_value + sv1.sum():.5f}")
    print(f"  Prognoze (modelis) = {float(predictor_core(instance_core.reshape(1,-1))[0]):.5f}")

    top_idx = np.argsort(-np.abs(sv1))[:config.SHAP_TOP_FEATURES]
    print(f"\n  Top-{config.SHAP_TOP_FEATURES} iezīmes šai dziesmai:")
    print(f"  {'Iezime':25s}  phi         virziens")
    print(f"  {'-'*55}")
    for i in top_idx:
        print(f"  {CORE_FEATURES[i]:25s}  {sv1[i]:+.5f}")

    # 6. Novērtēšanas metrikas
    print("NOVĒRTĒŠANAS METRIKAS")

    np.random.seed(0)
    fidelity    = evaluation.evaluate_fidelity(instance_core, predictor_core, sv1, expected_value)
    simplicity  = evaluation.evaluate_simplicity(sv1, CORE_FEATURES)
    consistency = evaluation.evaluate_consistency(sv1, sv2, CORE_FEATURES, k=10,
                                                   name_1="SHAP (seed 42)",
                                                   name_2="SHAP (seed 43)")
    robustness  = evaluation.evaluate_robustness(instance_core, predictor_core, explainer,
                                                  n_trials=config.EVAL_S_TRIALS,
                                                  sigma=0.01,
                                                  nsamples=config.SHAP_LOCAL_NSAMPLES)

    # 7. Kopsavilkums
    print("KOPSAVILKUMS")
    print()
    rows = [
        ("Fidelity | P_h(x)  [modelis]",    str(fidelity["fidelity_P_h"])),
        ("Fidelity | P_Mh(x) [SHAP E+Σphi]",str(fidelity["fidelity_P_Mh"])),
        ("Fidelity | Score (Eq.1)",          str(fidelity["fidelity_score"])),
        ("Fidelity | Decision (Eq.2)",       str(fidelity["fidelity_decision"])),
        ("Fidelity | Rezultāts",              fidelity["fidelity_score_verdict"]),
        ("──────────────────────────────────","────────────────────"),
        ("Simplicity | n_core_features",     str(simplicity["simplicity_n_features_total"])),
        ("Simplicity | tau=0.10",            str(simplicity["simplicity_tau_010"])),
        ("Simplicity | tau=0.05",            str(simplicity["simplicity_tau_005"])),
        ("Simplicity | tau=0.01",            str(simplicity["simplicity_tau_001"])),
        ("──────────────────────────────────","────────────────────"),
        ("Consistency | C(M1,M2) (Eq.4)",    str(consistency["consistency_score"])),
        ("Consistency | Rezultāts",           consistency["consistency_verdict"]),
        ("──────────────────────────────────","────────────────────"),
        ("Robustness | R_attr (Eq.5)",       str(robustness["robustness_score"])),
        ("Robustness | Std",                 str(robustness["robustness_std"])),
        ("Robustness | Rezultāts",            robustness["robustness_verdict"]),
    ]
    for label, val in rows:
        print(f"  {label:38s}: {val}")


if __name__ == "__main__":
    main()
