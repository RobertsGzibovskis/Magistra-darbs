# main.py
# Galvenais izpildes fails — savieno visus moduļus kopā

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import data
import recommender
import evaluation

# Izveido un saglabā LIME skaidrojuma grafiku.
def plot_lime_explanation(explanation, user_id, track_name, out_path):
    
    exp_list = explanation.as_list()
    # Sakārtojam, lai svarīgākās iezīmes būtu augšpusē
    exp_list.sort(key=lambda x: abs(x[1]), reverse=False) 
    
    features = [x[0] for x in exp_list]
    weights = [x[1] for x in exp_list]
    
    # Definējam krāsas: zaļš pozitīviem, sarkans negatīviem
    colors = ['green' if w > 0 else 'red' for w in weights]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, weights, color=colors, alpha=0.7)
    
    # Pievienojam vertikālu līniju pie 0
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.xlabel('Iezīmes ietekme uz ieteikuma punktiem (Weight)')
    plt.title(f'LIME Skaidrojums: {track_name}\n(Lietotājs {user_id})')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Pievienojam vērtības joslu galos precizitātei
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else width - 0.005
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                 f'{width:+.5f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"  [Vizuālais skaidrojums saglabāts]: {out_path}")
    plt.close()

# Atrod dziesmu, kuru sistēma visvairāk iesaka lietotājam user_idx, izslēdzot dziesmas, ko lietotājs jau ir klausījies
def find_top_recommendation(user_idx, plays_df, user_factors, item_factors):
    already_heard = set(
        plays_df[plays_df["user_idx"] == user_idx]["track_idx"].values
    )
     # CF vērtējumi visām dziesmām — ātra matricu reizināšana
    user_vec  = user_factors[user_idx]
    cf_scores = item_factors @ user_vec
    cf_norm   = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-9)

    mask = np.ones(len(cf_norm), dtype=bool)
    for idx in already_heard:
        mask[idx] = False
    cf_norm[~mask] = -1.0

    top_track = int(np.argmax(cf_norm))
    return top_track, float(cf_norm[top_track])


def _feature_group(feat: str) -> str:
    if feat.startswith("genre_"):
        return "žanrs"
    if any(x in feat for x in data.ERA_DEV_FEATURES):
        return "era-dev"
    if any(x in feat for x in data.BINARY_FEATURES):
        return "binārs"
    return "audio"


def main():
    print("LIME XAI — Mūzikas Ieteikumu Sistēma  (Spotify)")
    print(f"Izvades direktorija: {config.OUT}\n")

    # 1. Datu ielāde
    df = data.load_tracks(config.CSV, config.CSV_GENRES, config.CSV_BY_YEAR)

    # 2. Iezīmju matrica
    feature_matrix, feature_names, scaler = data.build_feature_matrix(df)

    # 3. Lietotāju simulācija
    plays_df, R, user_prefs = data.simulate_users(
        df, feature_matrix,
        n_users=config.N_USERS,
        seed=config.RANDOM_SEED
    )

    # 4. SVD modelis + LIME explainer (inicializēts BEZ žanriem)
    user_factors, item_factors = recommender.build_svd_model(
        R, n_components=config.SVD_COMPONENTS, seed=config.RANDOM_SEED
    )
   # LIME XAI
    make_predictor = recommender.make_scorer(
        user_factors, item_factors, feature_matrix, feature_names, plays_df
    )

    explainer, filtered_names = recommender.build_lime_explainer(
        feature_matrix, feature_names,
        n_background=config.LIME_BACKGROUND_N,
        seed=config.RANDOM_SEED
    )

    # 5. Dziesmas izvēle
    TARGET_USER = config.TARGET_USER
    standard_predictor = make_predictor(TARGET_USER)

    global TARGET_TRACK
    if config.TARGET_TRACK == -1:
        print(f"\nMeklē labāko ieteikumu lietotājam {TARGET_USER} ...")
        TARGET_TRACK, rec_score = find_top_recommendation(
            TARGET_USER, plays_df, user_factors, item_factors
        )
        config.TARGET_TRACK = TARGET_TRACK
        print(f"Ieteikts: track_idx={TARGET_TRACK}  score={rec_score:.4f}")
    else:
        TARGET_TRACK = config.TARGET_TRACK
        already_heard = set(plays_df[plays_df["user_idx"] == TARGET_USER]["track_idx"].values)
        if TARGET_TRACK in already_heard:
            print(f"\n[!] Dziesma {TARGET_TRACK} jau ir lietotāja {TARGET_USER} vēsturē.")
        rec_score = float(standard_predictor(feature_matrix[[TARGET_TRACK]])[0])
        print(f"\nNorādītā dziesma idx={TARGET_TRACK}  score={rec_score:.4f}")

    # Iesaldējam reālo CF komponenti target dziesmai stabila skaidrojuma iegūšanai
    user_vec  = user_factors[TARGET_USER]
    cf_scores = item_factors @ user_vec
    cf_norm   = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-9)
    frozen_cf = float(cf_norm[TARGET_TRACK])

    # Īpašais LIME prediktors ar iesaldētu CF
    lime_predictor = make_predictor(TARGET_USER, base_cf_score=frozen_cf)

    # 6. Dziesmas informācijas izvade vizualizācijai
    meta = df.iloc[TARGET_TRACK]
    active_genres = [g.replace("genre_", "").replace("_", " ")
                     for g in feature_names if g.startswith("genre_")
                     and feature_matrix[TARGET_TRACK, feature_names.index(g)] > 0.5]

    print(f"\nIzskaidro ieteikumu lietotājam {TARGET_USER}")
    print(f"  Preferences: ", end="")
    print("  |  ".join(f"{data.PREF_LABELS[p]}={user_prefs[p][TARGET_USER]}" for p in data.PREF_VALUES))
    print(f"  Dziesma     : {meta['name']}")
    print(f"  Izpildītājs : {meta['artist_name']}")
    print(f"  Gads        : {int(meta['year'])}")
    print(f"  Žanri       : {', '.join(active_genres) if active_genres else '(nav)' }")
    print(f"  Audio       : valence={meta['valence']:.2f}  dance={meta['danceability']:.2f}  energy={meta['energy']:.2f}  tempo={meta['tempo']:.0f}BPM")
    print(f"  Era-dev     : energy_vs_era={meta['energy_vs_era']:+.3f}  valence_vs_era={meta['valence_vs_era']:+.3f}")
    print(f"  Popularitāte: {meta['popularity']}")

    # Izgriežam instanci (tikai audio daļu bez žanriem priekš LIME)
    non_genre_indices = [i for i, name in enumerate(feature_names) if not name.startswith("genre_")]
    instance_filtered = feature_matrix[TARGET_TRACK, non_genre_indices]

    # 7. LIME skaidrojums #1
    np.random.seed(config.RANDOM_SEED)
    explanation = explainer.explain_instance(
        data_row    = instance_filtered,
        predict_fn  = lime_predictor,
        num_features= config.LIME_TOP_FEATURES,
        num_samples = config.LIME_NUM_SAMPLES
    )

    exp_list = sorted(explanation.as_list(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\nLIME Top-{config.LIME_TOP_FEATURES} Iezīmes (BEZ ŽANRIEM):")
    for feat, weight in exp_list:
        sign  = "+" if weight > 0 else "-"
        group = _feature_group(feat)
        print(f"  {sign}  [{group:8s}]  {feat:42s}  {weight:+.5f}")

    safe_track_name = "".join([c if c.isalnum() else "_" for c in meta['name']])
    out_img_name = f"lime_user{TARGET_USER}_{safe_track_name}.png"
    out_img_path = os.path.join(config.OUT, out_img_name)

    # Izsaucam vizualizācijas funkciju
    plot_lime_explanation(
        explanation=explanation,
        user_id=TARGET_USER,
        track_name=meta['name'],
        out_path=out_img_path
    )

    # LIME skaidrojums #2 (Konsistencei ar mainītu seed)
    np.random.seed(config.RANDOM_SEED + 1)
    explanation_2 = explainer.explain_instance(
        data_row    = instance_filtered,
        predict_fn  = lime_predictor,
        num_features= config.LIME_TOP_FEATURES,
        num_samples = config.LIME_NUM_SAMPLES
    )

    # 9. Novērtēšanas metrikas aprēķini
    print("\n")
    print("LIME XAI — NOVĒRTĒŠANAS METRIKAS")


    np.random.seed(0)
     # 1. Uzticamība (Fidelity)
    fidelity_results = evaluation.evaluate_fidelity(
        instance_filtered, lime_predictor, explanation, filtered_names
    )
# 2. Vienkāršība/sarežgītība (Simplicity)
    explanation_full = explainer.explain_instance(
        data_row    = instance_filtered,
        predict_fn  = lime_predictor,
        num_features= len(filtered_names),
        num_samples = config.LIME_NUM_SAMPLES
    )
    simplicity_results = evaluation.evaluate_simplicity(
        explanation_full, filtered_names, thresholds=(0.10, 0.05, 0.01)
    )

  # 3. Patstāvīgums (Consistency) divi LIME skaidrojumi tam pašam vienumam
    consistency_results = evaluation.evaluate_consistency(
        explanation_1=explanation,
        explanation_2=explanation_2,
        feature_names=filtered_names,
        k=min(10, len(filtered_names)),
        name_1="LIME (seed 42)",
        name_2="LIME (seed 43)"
    )
 # 4. Izturība (Robustness) sigma=0.01
    robustness_results = evaluation.evaluate_robustness(
        instance_filtered, lime_predictor, explainer, filtered_names,
        n_trials  =config.EVAL_S_TRIALS,
        sigma     =0.01,
        n_features=config.LIME_TOP_FEATURES,
        n_samples =config.LIME_NUM_SAMPLES
    )
    # Kopsavilkums
    print("\n")
    print("NOVĒRTĒŠANAS KOPSAVILKUMS")
    print()

    summary = {
        "Fidelity | P_h(x)  [modelis]"   : fidelity_results["fidelity_P_h"],
        "Fidelity | P_Mh(x) [surrogate]" : fidelity_results["fidelity_P_Mh"],
        "Fidelity | Score (Eq.1)"        : fidelity_results["fidelity_score"],
        "Fidelity | Decision (Eq.2)"     : fidelity_results["fidelity_decision"],
        "Fidelity | Rezultāts"            : fidelity_results["fidelity_score_verdict"],
        "Simplicity | tau=0.10"          : simplicity_results["simplicity_tau_010"],
        "Simplicity | tau=0.05"          : simplicity_results["simplicity_tau_005"],
        "Simplicity | tau=0.01"          : simplicity_results["simplicity_tau_001"],
        "Consistency | C(M1,M2)"         : consistency_results["consistency_score"],
        "Consistency | Rezultāts"         : consistency_results["consistency_verdict"],
        "Robustness | R_attr (Eq.5)"     : robustness_results["robustness_score"],
        "Robustness | Std"               : robustness_results["robustness_std"],
        "Robustness | Rezultāts"          : robustness_results["robustness_verdict"],
    }
    for k, v in summary.items():
        print(f"  {k:42s}: {v}")


if __name__ == "__main__":
    main()