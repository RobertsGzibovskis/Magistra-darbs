# main.py
# Galvenais izpildes fails — savieno visus moduļus kopā

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np

import config
import data
import recommender
import evaluation


def main():
    print("LIME XAI — Mūzikas Ieteikumu Sistēma")
    print(f"Izvades direktorija: {config.OUT}")

    # Datu ielāde un iezīmju matrica
    df = data.load_tracks(config.CSV)

    feature_matrix, feature_names, scaler = data.build_feature_matrix(df)


    # Lietotāju simulācija
    plays_df, R, user_genre, user_country = data.simulate_users(
        df, feature_matrix,
        n_users=config.N_USERS,
        seed=config.RANDOM_SEED
    )

    # SVD modelis
    user_factors, item_factors = recommender.build_svd_model(
        R,
        n_components=config.SVD_COMPONENTS,
        seed=config.RANDOM_SEED
    )

    # LIME XAI
    make_predictor = recommender.make_scorer(
        user_factors, item_factors,
        feature_matrix, feature_names, plays_df
    )

    explainer, background = recommender.build_lime_explainer(
        feature_matrix, feature_names,
        n_background=config.LIME_BACKGROUND_N,
        seed=config.RANDOM_SEED
    )

    # Lokālais LIME skaidrojums
    TARGET_USER  = config.TARGET_USER
    TARGET_TRACK = config.TARGET_TRACK

    meta = df.iloc[TARGET_TRACK]
    print(f"\nIzskaidro ieteikumu lietotājam {TARGET_USER}")
    print(f"  Žanra / valsts preference: "
          f"{user_genre[TARGET_USER]} / {user_country[TARGET_USER]}")
    print(f"  Dziesma  : {meta['track_name']}")
    print(f"  Izpildītājs: {meta['artist_name']}")
    print(f"  Žanrs    : {meta['genre']}  |  Valsts: {meta['country']}")
    print(f"  Audio    : dance={meta['danceability']:.2f}  "
          f"energy={meta['energy']:.2f}  "
          f"tempo={meta['tempo']:.0f}BPM  "
          f"loudness={meta['loudness']:.1f}dB")
    print(f"  Straumējumi: {meta['stream_count']:,}  |  "
          f"Popularitāte: {meta['popularity']}")

    predictor = make_predictor(TARGET_USER)
    instance  = feature_matrix[TARGET_TRACK]

    np.random.seed(config.RANDOM_SEED)
    explanation = explainer.explain_instance(
        data_row    = instance,
        predict_fn  = predictor,
        num_features= config.LIME_TOP_FEATURES,
        num_samples = config.LIME_NUM_SAMPLES
    )

    print(f"\nLIME Top-{config.LIME_TOP_FEATURES} Iezīmes:")
    for feat, weight in sorted(explanation.as_list(),
                                key=lambda x: abs(x[1]), reverse=True):
        
        sign = "+" if weight > 0 else "-"
        print(f"  {sign}{feat:45s}  {weight:+.5f}")

    # Batch izskaidrojumi (viens paraugs katram žanram)
        np.random.seed(config.RANDOM_SEED + 1)
    explanation_2 = explainer.explain_instance(
        data_row    = instance,
        predict_fn  = predictor,
        num_features= config.LIME_TOP_FEATURES,
        num_samples = config.LIME_NUM_SAMPLES
    )


    # Novērtēšanas metrikas 

    print("\n")
    print("LIME XAI — NOVERTESANAS METRIKAS")
 
    np.random.seed(0)
 
    # 1. Uziticamība (Fidelity)
    fidelity_results = evaluation.evaluate_fidelity(
        instance, predictor, explanation, feature_names,
    )
 
    # 2. Vienkāršība/sarežgītība (Simplicity)

    explanation_full = explainer.explain_instance(
        data_row    = instance,
        predict_fn  = predictor,
        num_features= len(feature_names),  
        num_samples = config.LIME_NUM_SAMPLES,
    )

    simplicity_results = evaluation.evaluate_simplicity(
        explanation_full, feature_names,
        thresholds=(0.10, 0.05, 0.01),
    )
 
    # 3. Patstāvīgums (Consistency)   divi LIME skaidrojumi tam pašam vienumam
    consistency_results = evaluation.evaluate_consistency(
        explanation_1=explanation,
        explanation_2=explanation_2,
        feature_names=feature_names,
        k=5,
        name_1="LIME (seed 42)",
        name_2="LIME (seed 43)",
    )
 
    # 4. Izturība (Robustness) sigma=0.01
    robustness_results = evaluation.evaluate_robustness(
        instance, predictor, explainer, feature_names,
        n_trials  =config.EVAL_S_TRIALS,
        sigma     =0.01,
        n_features=config.LIME_TOP_FEATURES,
        n_samples =config.LIME_NUM_SAMPLES,
    )
 
    # Kopsavilkums
 
    print("\n")
    print("NOVERTESANAS KOPSAVILKUMS")
    print("\n")
 
    summary = {
        
        "Fidelity | P_h(x)  [modelis]"   : fidelity_results["fidelity_P_h"],
        "Fidelity | P_Mh(x) [surrogate]" : fidelity_results["fidelity_P_Mh"],
        "Fidelity | Score"               : fidelity_results["fidelity_score"],
        "Fidelity | Decision"            : fidelity_results["fidelity_decision"],
        "Fidelity | Final"               : fidelity_results["fidelity_score_verdict"],
        
        "Simplicity | tau=0.10"          : simplicity_results["simplicity_tau_010"],
        "Simplicity | tau=0.05"          : simplicity_results["simplicity_tau_005"],
        "Simplicity | tau=0.01"          : simplicity_results["simplicity_tau_001"],
        
        "Consistency | C(M1,M2)"         : consistency_results["consistency_score"],
        "Consistency | Final"            : consistency_results["consistency_verdict"],
        
        "Robustness | R_attr"            : robustness_results["robustness_score"],
        "Robustness | Std"               : robustness_results["robustness_std"],
        "Robustness | Final"             : robustness_results["robustness_verdict"],
    }
    for k, v in summary.items():
        print(f"  {k:40s}: {v}")


if __name__ == "__main__":
    main()