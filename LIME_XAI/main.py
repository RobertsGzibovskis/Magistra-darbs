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

    N_TRACKS  = len(df)
    GENRES    = df["genre"].unique().tolist()
    COUNTRIES = df["country"].unique().tolist()

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
    genre_samples = {}
    for g in sorted(GENRES):
        idxs = df[df["genre"] == g].index.tolist()
        genre_samples[g] = idxs[0]

    print(f"\nBatch izskaidrojumu analīze — {len(genre_samples)} žanri …")
    batch_exps = {}
    for g, tidx in genre_samples.items():
        exp = explainer.explain_instance(
            feature_matrix[tidx], predictor,
            num_features=len(data.AUDIO_CORE),
            num_samples=600
        )
        batch_exps[g] = dict(exp.as_list())

    # Novērtēšanas metrikas D, R, F, S

    print("LIME XAI — NOVĒRTĒŠANAS METRIKAS  (D, R, F, S)")
   

    np.random.seed(0)

    d_results = evaluation.evaluate_D(
        instance, predictor, explainer, feature_names,
        delta     = config.EVAL_DELTA,
        n_samples = config.EVAL_D_SAMPLES,
        noise_std = config.EVAL_D_NOISE
    )

    r_results = evaluation.evaluate_R(
        explanation,
        c   = config.EVAL_R_THRESHOLD,
        lam = config.EVAL_LAMBDA
    )

    f_results = evaluation.evaluate_F(
        explanation, feature_names,
        f_threshold = config.EVAL_F_THRESHOLD,
        lam         = config.EVAL_LAMBDA
    )

    s_results = evaluation.evaluate_S(
        instance, predictor, explainer, feature_names,
        n_trials    = config.EVAL_S_TRIALS,
        noise_std   = config.EVAL_S_NOISE,
        lam         = config.EVAL_LAMBDA,
        num_features= 10,
        num_samples = 500
    )

    # Kopsavilkums 
    print("\n")
    print("NOVĒRTĒŠANAS KOPSAVILKUMS")

    summary = {
        "D — Melnā kaste nepieciešama" : "Jā" if d_results["Melnā kaste nepieciešama"] else "Nē",
        "D — Pb (melnā kaste R²)"      : d_results["Pb (melnās kastes R²)"],
        "D — Pt (caurspīdīgais R²)"    : d_results["Pt (caurspīdīgā modeļa R²)"],
        "R — Noteikumu skaits"         : r_results["Noteikumu skaits (m)"],
        "R — R vērtība (sods)"         : r_results["R = λ * L"],
        "F — Iezīmju skaits"           : f_results["Unikālās pamata iezīmes (f_used)"],
        "F — F vērtība (sods)"         : f_results["F = λ * max(0, f_used - f_thresh)"],
        "S — Vidējā Jaccard līdzība"   : s_results["Vidējā Jaccard līdzība"],
        "S — Vidējā Tanimoto līdzība"  : s_results["Vidējā Tanimoto līdzība"],
        "S — S_Jaccard"                : s_results["S_jaccard  = λ*(1 − mean_J)"],
        "S — S_Tanimoto"               : s_results["S_tanimoto = λ*(1 − mean_T)"],
        "S — Novērtējums"              : s_results["Stabilitātes novērtējums (J)"],
    }
    for k, v in summary.items():
        print(f"  {k:42s}: {v}")


if __name__ == "__main__":
    main()