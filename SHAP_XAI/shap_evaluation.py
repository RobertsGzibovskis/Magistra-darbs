# shap_evaluation.py

#   1. FIDELITY      — Score fidelity (Eq.1) + Decision fidelity (Eq.2)
#   2. SIMPLICITY    — Relatīvā sliekšņa retums (Eq.3), τ ∈ {0.10, 0.05, 0.01}
#   3. CONSISTENCY   — Dziļuma-vidēja Spearman pār top-k=5 (Eq.4)
#   4. ROBUSTNESS    — Vidējā L1 izmaiņa pie Gausa trokšņa σ=0.01 (Eq.5)

import numpy as np
from scipy.stats import spearmanr

import shap_config as config
from shap_recommender import CORE_FEATURES


def _attribution_vector(shap_values):
    return np.asarray(shap_values, dtype=float).flatten()


# 1. FIDELITY — Cik precīzi XAI metode reproducē modeļa uzvedību.

def evaluate_fidelity(instance_core, predictor_core, shap_values, expected_value):
 
    print("\nFIDELITY")

    P_h  = float(predictor_core(instance_core.reshape(1,-1))[0])
    P_Mh = float(expected_value + np.sum(shap_values))

    score_fidelity    = 1.0 - abs(P_h - P_Mh)
    y_h               = int(P_h  >= 0.5)
    y_Mh              = int(P_Mh >= 0.5)
    decision_fidelity = float(y_h == y_Mh)

    verdict = (
        "Augsta (>0.90)"     if score_fidelity > 0.90 else
        "Videja (0.70-0.90)" if score_fidelity > 0.70 else
        "Zema (<0.70)"
    )

    result = {
        "fidelity_P_h"           : round(P_h,             6),
        "fidelity_P_Mh"          : round(P_Mh,            6),
        "fidelity_abs_diff"      : round(abs(P_h - P_Mh), 6),
        "fidelity_score"         : round(score_fidelity,   4),
        "fidelity_y_h"           : y_h,
        "fidelity_y_Mh"          : y_Mh,
        "fidelity_decision"      : decision_fidelity,
        "fidelity_score_verdict" : verdict,
    }

    print(f"  P_h(x)  [modelis]        : {result['fidelity_P_h']}")
    print(f"  P_Mh(x) [SHAP E+Σphi]   : {result['fidelity_P_Mh']}")
    print(f"  |P_h - P_Mh|             : {result['fidelity_abs_diff']}")
    print(f"  Score Fidelity  (Eq.1)   : {result['fidelity_score']}")
    print(f"  y_h  (klase modelis)     : {result['fidelity_y_h']}")
    print(f"  y_Mh (klase SHAP)        : {result['fidelity_y_Mh']}")
    print(f"  Decision Fidelity (Eq.2) : {result['fidelity_decision']}  "
          f"({'sakrit' if decision_fidelity==1.0 else 'nesakrit'})")
    print(f"  Score verdikts           : {result['fidelity_score_verdict']}")
    return result


# 2. SIMPLICITY - Aprēķināts kā vidējais iezīmju skaits, kuru relatīvā nozīme pārsniedz slieksni τ
# Mazāks rezultāts = vienkāršāks skaidrojums (mazāk iezīmju).

def evaluate_simplicity(shap_values, feature_names=CORE_FEATURES,
                         thresholds=(0.10, 0.05, 0.01)):
    print("\nSIMPLICITY")
    print()

    phi     = _attribution_vector(shap_values)
    n_total  = len(phi)
    n_nonzero = int(np.sum(np.abs(phi) > 1e-9))
    max_abs   = float(np.max(np.abs(phi)))

    result = {
        "simplicity_n_features_total": n_total,
        "simplicity_n_nonzero":        n_nonzero,
    }

    print(f"  Kopejais iezimju skaits  : {n_total}")
    print(f"  Nenulles iezimes         : {n_nonzero}")
    print(f"  Max |phi|                : {max_abs:.6f}")
    print()

    # Top iezīmes
    fn = list(feature_names)
    top_idx = np.argsort(-np.abs(phi))[:min(8, n_total)]
    print(f"  {'Iezime':25s}  phi         relatīvi")
    print(f"  {'-'*50}")
    for i in top_idx:
        rel = abs(phi[i])/max_abs if max_abs > 1e-12 else 0.0
        print(f"  {fn[i]:25s}  {phi[i]:+.5f}   {rel:.3f}")
    print()

    tau_keys = {0.10: "simplicity_tau_010",
                0.05: "simplicity_tau_005",
                0.01: "simplicity_tau_001"}

    for tau in thresholds:
        count = int(np.sum(np.abs(phi)/max_abs > tau)) if max_abs > 1e-12 else 0
        key   = tau_keys.get(tau, f"simplicity_tau_{int(tau*100):03d}")
        result[key] = count
        print(f"  tau={tau:.2f}  nozimigo iezimju skaits = {count:2d} / {n_total}")

    print()
    print("  Interpretacija: mazaks skaitlis = vienkarsaks skaidrojums")
    return result


# 3. CONSISTENCY - Pārī salīdzina divu XAI metožu skaidrojumus, izmantojot Spearman ranga korelāciju progresīvos dziļumos n=1..k


def evaluate_consistency(shap_values_1, shap_values_2,
                          feature_names=CORE_FEATURES, k=5,
                          name_1="SHAP_run_1", name_2="SHAP_run_2"):

    print("\nCONSISTENCY")
    print(f"  Metode 1 : {name_1}")
    print(f"  Metode 2 : {name_2}")
    print(f"  k        : {k}")

    f1 = _attribution_vector(shap_values_1)
    f2 = _attribution_vector(shap_values_2)
    fn = list(feature_names)

    rank_1 = np.argsort(np.argsort(-np.abs(f1))) + 1
    rank_2 = np.argsort(np.argsort(-np.abs(f2))) + 1
    top_k_1 = np.argsort(-np.abs(f1))[:k]
    top_k_2 = np.argsort(-np.abs(f2))[:k]

    spearman_depths = []
    print(f"\n  Spearman pa dzilumiem n=1..{k}:")
    for n in range(1, k+1):
        combined = np.union1d(np.argsort(-np.abs(f1))[:n], np.argsort(-np.abs(f2))[:n])
        r1s, r2s = rank_1[combined], rank_2[combined]
        if len(combined) < 2 or r1s.std() < 1e-9 or r2s.std() < 1e-9:
            sp = 1.0 if np.array_equal(r1s, r2s) else 0.0
        else:
            sp = float(spearmanr(r1s, r2s).correlation)
            if np.isnan(sp): sp = 0.0
        spearman_depths.append(sp)
        print(f"    n={n}: Spearman = {sp:.4f}")

    score = float(np.mean(spearman_depths))
    verdict = (
        "Augsts (>0.70)"        if score > 0.70 else
        "Videjs (0.40–0.70)"    if score > 0.40 else
        "Zems (<0.40)"          if score > 0.00 else
        "Nav korelacijas vai inversa"
    )

    print(f"\n  Top-{k} iezimes (M1): {[fn[i] for i in top_k_1]}")
    print(f"  Top-{k} iezimes (M2): {[fn[i] for i in top_k_2]}")
    print(f"\n  Consistency C(M1,M2) (Eq.4) : {score:.4f}")
    print(f"  Interpretacija              : {verdict}")

    return {
        "consistency_score"           : round(score, 4),
        "consistency_spearman_depths" : [round(s, 4) for s in spearman_depths],
        "consistency_k"               : k,
        "consistency_verdict"         : verdict,
        "consistency_top_k_M1"        : [fn[i] for i in top_k_1],
        "consistency_top_k_M2"        : [fn[i] for i in top_k_2],
    }


# 4. ROBUSTNESS  — Skaidrojuma stabilitāte pret mazām ieejas perturbācijām.
# Aprēķināts kā vidējā L1 izmaiņa atribūciju vektoros pirms un pēc maza Gausa trokšņa

def evaluate_robustness(instance_core, predictor_core, explainer,
                         n_trials=config.EVAL_S_TRIALS,
                         sigma=0.01,
                         nsamples=1000):
    
    print("\nROBUSTNESS")
    print(f"  Perturbacijas n_trials : {n_trials}")
    print(f"  sigma                  : {sigma}")
    print(f"  SHAP nsamples/trial    : {nsamples}")

    sv_orig = _attribution_vector(
        explainer.shap_values(instance_core.reshape(1,-1),
                              nsamples=nsamples, silent=True)
    )

    l1_changes = []
    for t in range(n_trials):
        epsilon  = np.random.normal(0, sigma, size=instance_core.shape)
        x_noisy  = instance_core + epsilon
        sv_noisy = _attribution_vector(
            explainer.shap_values(x_noisy.reshape(1,-1),
                                  nsamples=nsamples, silent=True)
        )
        l1 = float(np.sum(np.abs(sv_orig - sv_noisy)))
        l1_changes.append(l1)
        print(f"  Trial {t+1:2d}: L1 = {l1:.6f}")

    score = float(np.mean(l1_changes))
    std   = float(np.std(l1_changes))

    verdict = (
        "Augsta izturiba (L1 < 0.01)"   if score < 0.01 else
        "Videja izturiba (0.01–0.05)"   if score < 0.05 else
        "Zema izturiba   (L1 > 0.05)"
    )

    print(f"\n  R_attr (videja L1, Eq.5) : {score:.6f}")
    print(f"  Std L1                   : {std:.6f}")
    print(f"  Interpretacija           : {verdict}")
    print("  (Mazaka vertiba = augstaka izturiba)")

    return {
        "robustness_score"     : round(score, 6),
        "robustness_std"       : round(std,   6),
        "robustness_l1_trials" : [round(v, 6) for v in l1_changes],
        "robustness_sigma"     : sigma,
        "robustness_n_trials"  : n_trials,
        "robustness_verdict"   : verdict,
    }
