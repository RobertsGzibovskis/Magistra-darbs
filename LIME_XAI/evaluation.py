# evaluation.py

#   1. FIDELITY      — Score fidelity (Eq.1) + Decision fidelity (Eq.2)
#   2. SIMPLICITY    — Relative-threshold sparsity (Eq.3), τ ∈ {0.10, 0.05, 0.01}
#   3. CONSISTENCY   — Depth-averaged Spearman over top-k=5 (Eq.4)
#   4. ROBUSTNESS    — Mean L1 change under Gaussian noise σ=0.01 (Eq.5)

import numpy as np
from scipy.stats import spearmanr

import config

# Izvelk LIME surogāta svaru vektoru w un konstanti w0.
# Palīgfunkcijas 
def _get_surrogate_weights_and_intercept(explanation, feature_names):
    intercept_raw = getattr(explanation, "intercept", {})
    if isinstance(intercept_raw, dict):
        w0 = float(list(intercept_raw.values())[0]) if intercept_raw else 0.0
    else:
        w0 = float(intercept_raw) if intercept_raw else 0.0

    w = np.zeros(len(feature_names))
    for feat_str, val in explanation.as_list():
        for j, fn in enumerate(feature_names):
            if fn in feat_str:
                w[j] = val
                break
    return w, w0

# g(x) = w0 + x @ w  — surogāta modeļa prognoze vienam vektoram.
def _surrogate_score(w, w0, x):
    return float(w0 + x @ w)
# δ(a,b) = 1 ja a==b, citādi 0.
def _kronecker_delta(a, b):
    return 1 if a == b else 0

# Atgriež pilnu atribūtu vektoru no LIME skaidrojuma. Neizmantotās iezīmes ir ar vērtējumu 0
def _attribution_vector(explanation, feature_names):
    w, _ = _get_surrogate_weights_and_intercept(explanation, feature_names)
    return w



# 1. FIDELITY — Cik precīzi XAI metode reproducē modeļa uzvedību.


def evaluate_fidelity(instance, predictor, explanation, feature_names):
   
    print("\n")
    print("FIDELITY")

    w, w0 = _get_surrogate_weights_and_intercept(explanation, feature_names)

    # P_h(x) — modeļa prognoze (score)
    P_h = float(predictor(instance.reshape(1, -1))[0])

    # P_Mh(x) — surogāta rekonstrukcija
    P_Mh = _surrogate_score(w, w0, instance)

    # Eq. 1 — Score fidelity: 1 - |P_h(x) - P_Mh(x)|
    score_fidelity = 1.0 - abs(P_h - P_Mh)

    # Eq. 2 — Decision fidelity: δ(ŷ_h, ŷ_Mh)
    # Klases etiķetes: threshold 0.5 prognozei
    y_h  = int(P_h  >= 0.5)
    y_Mh = int(P_Mh >= 0.5)
    decision_fidelity = float(_kronecker_delta(y_h, y_Mh))

    verdict_score = (
        "Augsta (>0.90)"    if score_fidelity > 0.90 else
        "Videja (0.70-0.90)" if score_fidelity > 0.70 else
        "Zema (<0.70)"
    )

    result = {
        "fidelity_P_h"             : round(P_h, 6),
        "fidelity_P_Mh"            : round(P_Mh, 6),
        "fidelity_abs_diff"        : round(abs(P_h - P_Mh), 6),
        "fidelity_score"           : round(score_fidelity, 4),
        "fidelity_y_h"             : y_h,
        "fidelity_y_Mh"            : y_Mh,
        "fidelity_decision"        : decision_fidelity,
        "fidelity_score_verdict"   : verdict_score,
    }

    print(f"  P_h(x)  [modelis]        : {result['fidelity_P_h']}")
    print(f"  P_Mh(x) [surrogāts]      : {result['fidelity_P_Mh']}")
    print(f"  |P_h - P_Mh|             : {result['fidelity_abs_diff']}")
    print(f"  Score Fidelity  (Eq.1)   : {result['fidelity_score']}")
    print(f"  y_h  (klase modelis)     : {result['fidelity_y_h']}")
    print(f"  y_Mh (klase surrogāts)   : {result['fidelity_y_Mh']}")
    print(f"  Decision Fidelity (Eq.2) : {result['fidelity_decision']}  "
          f"({'sakrit' if decision_fidelity == 1.0 else 'nesakrit'})")
    print(f"  Score verdikts           : {result['fidelity_score_verdict']}")
    return result


# 2. SIMPLICITY - Aprēķināts kā vidējais iezīmju skaits, kuru relatīvā nozīme pārsniedz slieksni τ
# Mazāks rezultāts = vienkāršāks skaidrojums (mazāk iezīmju).

def evaluate_simplicity(explanation_full, feature_names,
                        thresholds=(0.10, 0.05, 0.01)):

    print("\n")
    print("SIMPLICITY")
    print("\n")

    # Atribūciju vektors f_i ∈ R^|F|
    f_i = _attribution_vector(explanation_full, feature_names)
    n_total = len(f_i)
    n_nonzero = int(np.sum(np.abs(f_i) > 1e-12))
    max_abs = np.max(np.abs(f_i))
 
    result = {"simplicity_n_features_total": n_total,
              "simplicity_n_nonzero": n_nonzero}
 
    print(f"  Kopejais iezimju skaits  : {n_total}")
    print(f"  Nenulles iezimes         : {n_nonzero}")
    print(f"  Max |atribucija|         : {max_abs:.6f}")
    print()

 
    tau_keys = {0.10: "simplicity_tau_010",
                0.05: "simplicity_tau_005",
                0.01: "simplicity_tau_001"}
    # Robežvērtības
    for tau in thresholds:
        if max_abs > 1e-12:
            relative = np.abs(f_i) / max_abs
            count = int(np.sum(relative > tau))
        else:
            count = 0
 
        key = tau_keys.get(tau, f"simplicity_tau_{int(tau*100):03d}")
        result[key] = count
        print(f"  tau={tau:.2f}  iezimju skaits = {count:3d} / {n_total}")
 
    print()
    print("  Interpretacija: mazaks skaitlis = vienkarsaks skaidrojums")
    return result

# 3. CONSISTENCY - Pārī salīdzina divu XAI metožu skaidrojumus, izmantojot Spearman ranga korelāciju progresīvos dziļumos n=1..k


def evaluate_consistency(explanation_1, explanation_2,
                         feature_names, k=5,
                         name_1="LIME_run_1", name_2="LIME_run_2"):

    print("\n")
    print("CONSISTENCY")
    print(f"  Metode 1 : {name_1}")
    print(f"  Metode 2 : {name_2}")
    print(f"  k        : {k}")

    f1 = _attribution_vector(explanation_1, feature_names)
    f2 = _attribution_vector(explanation_2, feature_names)

    # Rangu vektori: lielāka |atribūcija| -  zemāks rangs (1=svarīgākais)
    rank_1 = np.argsort(np.argsort(-np.abs(f1))) + 1   # 1-based ranks
    rank_2 = np.argsort(np.argsort(-np.abs(f2))) + 1

    # top-n iezīmju indeksi katrai metodei
    top_k_idx_1 = np.argsort(-np.abs(f1))[:k]
    top_k_idx_2 = np.argsort(-np.abs(f2))[:k]

    spearman_per_depth = []

    print(f"\n  Spearman pa dzilumiem n=1..{k}:")
    for n in range(1, k + 1):
        # top-n iezīmju rangi no abām metodēm (uz kopīgās iezīmju kopas)
        # Ņemam top-n no M1 un vērtējam rangu vektoru abus
        top_n_1 = np.argsort(-np.abs(f1))[:n]
        top_n_2 = np.argsort(-np.abs(f2))[:n]
        combined = np.union1d(top_n_1, top_n_2)

        r1_sub = rank_1[combined]
        r2_sub = rank_2[combined]

        if len(combined) < 2 or r1_sub.std() < 1e-9 or r2_sub.std() < 1e-9:
            sp = 1.0 if np.array_equal(r1_sub, r2_sub) else 0.0
        else:
            sp = float(spearmanr(r1_sub, r2_sub).correlation)
            if np.isnan(sp):
                sp = 0.0

        spearman_per_depth.append(sp)
        print(f"    n={n}: Spearman = {sp:.4f}")

    consistency_score = float(np.mean(spearman_per_depth))

    verdict = (
        "Aaugsts rezultāts (>0.70)"       if consistency_score > 0.70 else
        "Videjs rezultāts (0.40-0.70)"   if consistency_score > 0.40 else
        "Zems rezultāts (<0.40)"         if consistency_score > 0.00 else
        "Nav korelacijas vai inversa"
    )

    print(f"\n  Top-{k} iezimes (M1): "
          f"{[feature_names[i][:20] for i in top_k_idx_1]}")
    print(f"  Top-{k} iezimes (M2): "
          f"{[feature_names[i][:20] for i in top_k_idx_2]}")
    print(f"\n  Consistency C(M1,M2) (Eq.4) : {consistency_score:.4f}")
    print(f"  Interpretacija              : {verdict}")

    result = {
        "consistency_score"          : round(consistency_score, 4),
        "consistency_spearman_depths": [round(s, 4) for s in spearman_per_depth],
        "consistency_k"              : k,
        "consistency_verdict"        : verdict,
        "consistency_top_k_M1"       : [feature_names[i] for i in top_k_idx_1],
        "consistency_top_k_M2"       : [feature_names[i] for i in top_k_idx_2],
    }
    return result


# 4. ROBUSTNESS  — Skaidrojuma stabilitāte pret mazām ieejas perturbācijām.
# Aprēķināts kā vidējā L1 izmaiņa atribūciju vektoros pirms un pēc maza Gausa trokšņa


def evaluate_robustness(instance, predictor, explainer, feature_names,
                        n_trials=config.EVAL_S_TRIALS,
                        sigma=0.01,
                        n_features=config.LIME_TOP_FEATURES,
                        n_samples=config.LIME_NUM_SAMPLES):
    print("\n")
    print("ROBUSTNESS")
    print(f"  Perturbacijas n_trials : {n_trials}")
    print(f"  sigma                  : {sigma}")

    # Oriģinālais atribūciju vektors E(M_h, x)
    exp_orig = explainer.explain_instance(
        instance, predictor,
        num_features=n_features,
        num_samples=n_samples
    )
    e_orig = _attribution_vector(exp_orig, feature_names)

    l1_changes = []

    for t in range(n_trials):
        # ε ~ N(0, σ²)
        epsilon   = np.random.normal(0, sigma, size=instance.shape)
        x_noisy   = instance + epsilon

        # E(M_h, x + ε) — atribūciju vektors perturbētajai instancei
        exp_noisy = explainer.explain_instance(
            x_noisy, predictor,
            num_features=n_features,
            num_samples=n_samples
        )
        e_noisy = _attribution_vector(exp_noisy, feature_names)

        l1 = float(np.sum(np.abs(e_orig - e_noisy)))
        l1_changes.append(l1)

        print(f"  Trial {t+1:2d}: L1 = {l1:.6f}")

    robustness_score = float(np.mean(l1_changes))
    robustness_std   = float(np.std(l1_changes))

    verdict = (
        "Augsta izturība (L1 < 0.01)"    if robustness_score < 0.01 else
        "Videja izturība (0.01 - 0.05)"  if robustness_score < 0.05 else
        "Zema izturība   (L1 > 0.05)"
    )

    print(f"\n  R_attr (videja L1, Eq.5) : {robustness_score:.6f}")
    print(f"  Std L1                   : {robustness_std:.6f}")
    print(f"  Interpretacija           : {verdict}")
    print("  (Mazaka vertiba = augstaka izturība)")

    result = {
        "robustness_score"     : round(robustness_score, 6),
        "robustness_std"       : round(robustness_std,   6),
        "robustness_l1_trials" : [round(v, 6) for v in l1_changes],
        "robustness_sigma"     : sigma,
        "robustness_n_trials"  : n_trials,
        "robustness_verdict"   : verdict,
    }
    return result