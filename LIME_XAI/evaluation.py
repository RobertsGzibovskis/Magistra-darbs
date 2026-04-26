# evaluation.py
# LIME XAI novērtēšanas metrikas: D, R, F, S
#
# D — Melnās kastes nepieciešamība  (Black-box Necessity)
# R — Noteikumu sarežģītība         (Rule complexity)
# F — Iezīmju sarežģītība           (Feature complexity)
# S — Stabilitāte                   (Stability — Jaccard & Tanimoto)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import config

# Palīgfunkcijas

  # Izveido perturbētus paraugus ap doto punktu.
  # Gausa troksnis (std=noise_std), apgriezts intervālā [0, 1].
def _perturb_instance(instance: np.ndarray, n_samples: int = 200,
                      noise_std: float = 0.05) -> np.ndarray:
    
    noise = np.random.normal(0, noise_std, size=(n_samples, len(instance)))
    return np.clip(instance + noise, 0, 1)

#  Atgriež to LIME iezīmju kopu, kuru |svars| > threshold.
def _feature_set(lime_dict: dict, threshold: float = 0.0) -> set:
    return {k for k, v in lime_dict.items() if abs(v) > threshold}

# Jaccard līdzība starp divām kopām.
# J(A, B) = |A ∩ B| / |A ∪ B|
def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 0.0

# Tanimoto (ģeneralizētais Jaccard) koeficients nepārtrauktiem vektoriem.
# T(a, b) = dot(a,b) / (||a||² + ||b||² − dot(a,b))

def _tanimoto(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
   
    dot   = np.dot(vec_a, vec_b)
    denom = np.dot(vec_a, vec_a) + np.dot(vec_b, vec_b) - dot
    return float(dot / denom) if denom != 0 else 1.0



# D — Melnās kastes nepieciešamība
# Pb  = melnās kastes modeļa sniegums (R²)
# Pt  = caurspīdīgā (LIME surogāts) modeļa sniegums (R²)
# δ   = pieļaujamais snieguma samazinājums
# Lēmums: ja Pb − δ ≤ Pt → melnā kaste NAV nepieciešama

def evaluate_D(instance, predictor, explainer, feature_names,
               delta=config.EVAL_DELTA,
               n_samples=config.EVAL_D_SAMPLES,
               noise_std=config.EVAL_D_NOISE) -> dict:
    print("\n")
    print("D METRIKA — Melnās kastes nepieciešamība")
  
    X_local = _perturb_instance(instance, n_samples=n_samples,
                                noise_std=noise_std)
    y_black = predictor(X_local)

    split = int(0.8 * n_samples)
    X_tr, X_te = X_local[:split], X_local[split:]
    y_tr, y_te = y_black[:split], y_black[split:]

    bb_model = Ridge(alpha=1.0)
    bb_model.fit(X_tr, y_tr)
    Pb = max(0.0, r2_score(y_te, bb_model.predict(X_te)))

    exp = explainer.explain_instance(instance, predictor,
                                     num_features=10, num_samples=500)
    lime_feat_idx = []
    for feat_name, _ in exp.as_list():
        for j, fn in enumerate(feature_names):
            if fn in feat_name:
                lime_feat_idx.append(j)
                break
    lime_feat_idx = list(set(lime_feat_idx))

    if lime_feat_idx:
        trans_model = Ridge(alpha=1.0)
        trans_model.fit(X_tr[:, lime_feat_idx], y_tr)
        Pt = max(0.0, r2_score(y_te, trans_model.predict(X_te[:, lime_feat_idx])))
    else:
        Pt = 0.0

    necessary = (Pb - delta) > Pt
    verdict   = ("Jā — melnā kaste ir nepieciešama (Pb − δ > Pt)"
                 if necessary else
                 "Nē — pietiek ar caurspīdīgo modeli (Pb − δ ≤ Pt)")

    result = {
        "Pb (melnās kastes R²)"      : round(Pb, 4),
        "Pt (caurspīdīgā modeļa R²)" : round(Pt, 4),
        "δ (tolerance)"              : delta,
        "Pb - δ"                     : round(Pb - delta, 4),
        "Melnā kaste nepieciešama"  : necessary,
        "Lēmums"                     : verdict,
    }
    for k, v in result.items():
        print(f"  {k:40s}: {v}")
    return result



# R — Noteikumu sarežģītība

#  R = λ * L,  L = max(0, size(m) − c)
#  size(m) = noteikumu skaits LIME skaidrojumā
#  c       = pieļaujamais noteikumu skaits
#  Jo mazāks R, jo saprotamāks skaidrojums.
  
def evaluate_R(exp,
               c=config.EVAL_R_THRESHOLD,
               lam=config.EVAL_LAMBDA) -> dict:
  
    print("\n")
    print("R METRIKA — Noteikumu sarežģītība")
    

    rules   = exp.as_list()
    m       = len(rules)
    L       = max(0, m - c)
    R_score = lam * L

    print(f"\n  Noteikumu saraksts (m={m}):")
    for i, (feat, weight) in enumerate(
            sorted(rules, key=lambda x: abs(x[1]), reverse=True), 1):
        print(f"    {i:2d}. {feat:45s}  {weight:+.5f}")

    result = {
        "Noteikumu skaits (m)"        : m,
        "Pieļaujamais skaits (c)"     : c,
        "Pārsniegums L = max(0, m-c)" : L,
        "λ (soda koeficients)"        : lam,
        "R = λ * L"                   : round(R_score, 4),
        "Interpretācija"              : (
            "Skaidrojums ir pietiekami kompakts"
            if R_score == 0 else
            f"Skaidrojums ir pārāk sarežģīts — {L} lieki noteikumi"
        ),
    }

    print()
    for k, v in result.items():
        print(f"  {k:40s}: {v}")

    return result


# F — Iezīmju sarežģītība
#  F = λ * max(0, f_used − f_threshold)
#  f_used = unikālo pamata iezīmju skaits skaidrojumā
#  Jo mazāk iezīmju, jo skaidrāks skaidrojums.
   
def evaluate_F(exp, feature_names,
               f_threshold=config.EVAL_F_THRESHOLD,
               lam=config.EVAL_LAMBDA) -> dict:
    
    print("\n")
    print("F METRIKA — Iezīmju sarežģītība")
    lime_dict = dict(exp.as_list())

    used_base = set()
    for lime_key in lime_dict:
        for fname in feature_names:
            if fname in lime_key:
                used_base.add(fname)
                break

    f_used   = len(used_base)
    excess   = max(0, f_used - f_threshold)
    F_score  = lam * excess

    pos_feats = [(k, v) for k, v in lime_dict.items() if v > 0]
    neg_feats = [(k, v) for k, v in lime_dict.items() if v < 0]

    print(f"\n  Pozitīvās iezīmes:")
    for k, v in sorted(pos_feats, key=lambda x: x[1], reverse=True):
        print(f"    {k:45s}  +{v:.5f}")

    print(f"\n  Negatīvās iezīmes:")
    for k, v in sorted(neg_feats, key=lambda x: x[1]):
        print(f"    {k:45s}  {v:.5f}")

    result = {
        "Unikālās pamata iezīmes (f_used)"  : f_used,
        "Izmantotās iezīmes"                : sorted(used_base),
        "Pieļaujamais skaits (f_threshold)" : f_threshold,
        "Pārsniegums"                       : excess,
        "λ (soda koeficients)"              : lam,
        "F = λ * max(0, f_used - f_thresh)" : round(F_score, 4),
        "Pozitīvo iezīmju skaits"           : len(pos_feats),
        "Negatīvo iezīmju skaits"           : len(neg_feats),
        "Interpretācija"                    : (
            "Skaidrojums izmanto pieļaujamu iezīmju skaitu"
            if F_score == 0 else
            f"Pārāk daudz iezīmju — {excess} virs sliekšņa"
        ),
    }
    print()
    for k, v in result.items():
        if k != "Izmantotās iezīmes":
            print(f"  {k:45s}: {v}")
    print(f"  {'Izmantotās iezīmes':45s}: {sorted(used_base)}")
    return result

# S metrika — skaidrojumu stabilitāte pret troksni.
#  S = λ * (1 − similarity)
#  similarity — Jaccard (iezīmju kopas) vai Tanimoto (svaru vektori)
#  Maza S → stabils skaidrojums.

def evaluate_S(instance, predictor, explainer, feature_names,
               n_trials=config.EVAL_S_TRIALS,
               noise_std=config.EVAL_S_NOISE,
               lam=config.EVAL_LAMBDA,
               num_features=10,
               num_samples=500) -> dict:
  
    print("\n")
    print("S METRIKA — Stabilitāte")
   
    print(f"  Perturbācijas: {n_trials}  |  Trokšņa σ: {noise_std}")

    exp_orig  = explainer.explain_instance(instance, predictor,
                                           num_features=num_features,
                                           num_samples=num_samples)
    orig_dict = dict(exp_orig.as_list())
    orig_set  = _feature_set(orig_dict)

    jaccard_scores, tanimoto_scores, trial_results = [], [], []

    for trial in range(n_trials):
        noise      = np.random.normal(0, noise_std, size=instance.shape)
        inst_noisy = np.clip(instance + noise, 0, 1)

        exp_noisy  = explainer.explain_instance(inst_noisy, predictor,
                                                num_features=num_features,
                                                num_samples=num_samples)
        noisy_dict = dict(exp_noisy.as_list())
        noisy_set  = _feature_set(noisy_dict)

        j = _jaccard(orig_set, noisy_set)

        combined_keys = sorted(orig_set | noisy_set)
        v_orig  = np.array([orig_dict.get(k, 0.0)  for k in combined_keys])
        v_noisy = np.array([noisy_dict.get(k, 0.0) for k in combined_keys])
        t = _tanimoto(v_orig, v_noisy)

        jaccard_scores.append(j)
        tanimoto_scores.append(t)
        trial_results.append({
            "Mēģinājums"     : trial + 1,
            "Jaccard"        : round(j, 4),
            "Tanimoto"       : round(t, 4),
            "Kopīgas iezīmes": sorted(orig_set & noisy_set),
            "Jaunās iezīmes" : sorted(noisy_set - orig_set),
            "Pazudušās"      : sorted(orig_set - noisy_set),
        })
        print(f"  Mēģinājums {trial+1}: Jaccard={j:.4f}  Tanimoto={t:.4f}  "
              f"kopīgas={len(orig_set & noisy_set)}/{len(orig_set | noisy_set)}")

    mean_j = np.mean(jaccard_scores)
    mean_t = np.mean(tanimoto_scores)

    S_jaccard  = lam * (1 - mean_j)
    S_tanimoto = lam * (1 - mean_t)

    def _label(s):
        if s < 0.10:  return "Stabils"
        if s < 0.25:  return "Pietiekami stabils"
        if s < 0.50:  return "Nestabils"
        return              "Ļoti nestabils — skaidrojumi mainās ievērojami"

    result = {
        "Vidējā Jaccard līdzība"         : round(mean_j, 4),
        "Vidējā Tanimoto līdzība"        : round(mean_t, 4),
        "Jaccard standartnovirze"        : round(np.std(jaccard_scores),  4),
        "Tanimoto standartnovirze"       : round(np.std(tanimoto_scores), 4),
        "S_jaccard  = λ*(1 − mean_J)"   : round(S_jaccard,  4),
        "S_tanimoto = λ*(1 − mean_T)"   : round(S_tanimoto, 4),
        "Stabilitātes novērtējums (J)"  : _label(S_jaccard),
        "Stabilitātes novērtējums (T)"  : _label(S_tanimoto),
        "Mēģinājumu dati"               : trial_results,
        "noise_std"                     : noise_std,
    }
    print()
    for k, v in result.items():
        if k not in ("Mēģinājumu dati", "noise_std"):
            print(f"  {k:45s}: {v}")
    return result
