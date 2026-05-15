import warnings
warnings.filterwarnings("ignore")

import sys, os, io, contextlib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "LIME_XAI"))
sys.path.append(os.path.join(BASE_DIR, "SHAP_XAI"))

from LIME_XAI import config       as lime_config
from LIME_XAI import data         as lime_data
from LIME_XAI import recommender  as lime_recommender
from LIME_XAI import evaluation   as lime_evaluation

from SHAP_XAI import shap_config  as shap_config
sys.modules["config"] = shap_config
from SHAP_XAI import shap_data        as shap_data_mod
from SHAP_XAI import shap_recommender as shap_rec
from SHAP_XAI import shap_evaluation  as shap_eval
from SHAP_XAI.shap_recommender import CORE_FEATURES as SHAP_CORE_FEATURES
sys.modules["config"] = lime_config

# Konfigurācija 

EVAL_USERS = list(range(1, 51))
EVAL_TOP_TRACKS = 3
EVAL_RUNS       = 4
OUT_DIR         = lime_config.OUT

METRIC_COLS = [
    "fidelity_score", "fidelity_decision", "fidelity_abs_diff",
    "simplicity_tau_010", "simplicity_tau_005", "simplicity_tau_001",
    "consistency_score", "robustness_score",
]
METRIC_LABELS = {
    "fidelity_score"    : "Fidelity Score",
    "fidelity_decision" : "Decision Fidelity",
    "fidelity_abs_diff" : "Fidelity |P_h - P_Mh|",
    "simplicity_tau_010": "Simplicity  τ=0.10",
    "simplicity_tau_005": "Simplicity  τ=0.05",
    "simplicity_tau_001": "Simplicity  τ=0.01",
    "consistency_score" : "Consistency C(M1,M2)",
    "robustness_score"  : "Robustness  R_attr",
}
LOWER_IS_BETTER = {
    "robustness_score", "fidelity_abs_diff",
    "simplicity_tau_010", "simplicity_tau_005", "simplicity_tau_001",
}

# Palīgfunkcijas

def _s(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)

def top_n(uid, n, plays, uf, if_):
    heard = set(plays[plays["user_idx"] == uid]["track_idx"].values)
    cf = if_ @ uf[uid]
    cf = (cf - cf.min()) / (cf.max() - cf.min() + 1e-9)
    for i in heard:
        cf[i] = -1.0
    return [(int(t), float(cf[t])) for t in np.argsort(-cf)[:n]]

def cf_at(uid, tid, uf, if_):
    cf = if_ @ uf[uid]
    cf = (cf - cf.min()) / (cf.max() - cf.min() + 1e-9)
    return float(cf[tid])

def lime_vec(tid, fm, fn):
    return fm[tid, [i for i, n in enumerate(fn) if not n.startswith("genre_")]]

# Datu ielāde

def load_state():
    df_l = _s(lime_data.load_tracks, lime_config.CSV, lime_config.CSV_GENRES, lime_config.CSV_BY_YEAR)
    fm_l, fn_l, _ = _s(lime_data.build_feature_matrix, df_l)
    plays, R, _   = _s(lime_data.simulate_users, df_l, fm_l, n_users=lime_config.N_USERS, seed=lime_config.RANDOM_SEED)
    uf_l, if_l    = _s(lime_recommender.build_svd_model, R, n_components=lime_config.SVD_COMPONENTS, seed=lime_config.RANDOM_SEED)
    scorer_l      = lime_recommender.make_scorer(uf_l, if_l, fm_l, fn_l, plays)
    expl_l, fn_f  = _s(lime_recommender.build_lime_explainer, fm_l, fn_l, n_background=lime_config.LIME_BACKGROUND_N, seed=lime_config.RANDOM_SEED)

    sys.modules["config"] = shap_config
    df_s = _s(shap_data_mod.load_tracks, shap_config.CSV)
    fm_s, fn_s, _ = shap_data_mod.build_feature_matrix(df_s)
    plays_s, R_s, _ = _s(shap_data_mod.simulate_users, df_s, fm_s, n_users=shap_config.N_USERS, seed=shap_config.RANDOM_SEED)
    uf_s, if_s    = _s(shap_rec.build_svd_model, R_s, n_components=shap_config.SVD_COMPONENTS, seed=shap_config.RANDOM_SEED)
    scorer_s      = shap_rec.make_scorer(uf_s, if_s, fm_s, fn_s, plays_s)
    sys.modules["config"] = lime_config

    return dict(fm_l=fm_l, fn_l=fn_l, plays=plays, uf_l=uf_l, if_l=if_l,
                scorer_l=scorer_l, expl_l=expl_l, fn_f=fn_f,
                fm_s=fm_s, fn_s=fn_s, uf_s=uf_s, if_s=if_s, scorer_s=scorer_s)

# Metriku aprēķināšana

def run_lime(uid, tid, lcf, st, s1, s2):
    lime_config.TARGET_TRACK = tid
    pred = st["scorer_l"](uid, base_cf_score=lcf)
    inst = lime_vec(tid, st["fm_l"], st["fn_l"])
    e    = st["expl_l"]
    fn   = st["fn_f"]
    ef   = _s(e.explain_instance, inst, pred, num_features=len(fn),                    num_samples=lime_config.LIME_NUM_SAMPLES)
    np.random.seed(s1); e1 = _s(e.explain_instance, inst, pred, num_features=lime_config.LIME_TOP_FEATURES, num_samples=lime_config.LIME_NUM_SAMPLES)
    np.random.seed(s2); e2 = _s(e.explain_instance, inst, pred, num_features=lime_config.LIME_TOP_FEATURES, num_samples=lime_config.LIME_NUM_SAMPLES)
    fid  = _s(lime_evaluation.evaluate_fidelity,    inst, pred, e1, fn)
    simp = _s(lime_evaluation.evaluate_simplicity,  ef, fn)
    cons = _s(lime_evaluation.evaluate_consistency, e1, e2, fn, k=10)
    rob  = _s(lime_evaluation.evaluate_robustness,  inst, pred, e, fn,
              n_trials=lime_config.EVAL_S_TRIALS, sigma=0.01,
              n_features=lime_config.LIME_TOP_FEATURES, n_samples=lime_config.LIME_NUM_SAMPLES)
    return fid, simp, cons, rob

def run_shap(uid, tid, st, s1, s2):
    sys.modules["config"] = shap_config
    expl, _, _, fm_core, ev, pred_core = _s(shap_rec.build_shap_explainer,
        st["fm_s"], st["fn_s"], st["scorer_s"](uid),
        n_background=shap_config.SHAP_BACKGROUND_N, seed=shap_config.RANDOM_SEED)
    inst = fm_core[tid]
    np.random.seed(s1); sv1 = np.asarray(expl.shap_values(inst.reshape(1,-1), nsamples=shap_config.SHAP_LOCAL_NSAMPLES, silent=True)).flatten()
    np.random.seed(s2); sv2 = np.asarray(expl.shap_values(inst.reshape(1,-1), nsamples=shap_config.SHAP_LOCAL_NSAMPLES, silent=True)).flatten()
    fid  = _s(shap_eval.evaluate_fidelity,    inst, pred_core, sv1, ev)
    simp = _s(shap_eval.evaluate_simplicity,  sv1, SHAP_CORE_FEATURES)
    cons = _s(shap_eval.evaluate_consistency, sv1, sv2, SHAP_CORE_FEATURES, k=10)
    rob  = _s(shap_eval.evaluate_robustness,  inst, pred_core, expl,
              n_trials=shap_config.EVAL_S_TRIALS, sigma=0.01, nsamples=shap_config.SHAP_LOCAL_NSAMPLES)
    sys.modules["config"] = lime_config
    return fid, simp, cons, rob

# Galvenā evaluation funkcija

def run_combined_evaluation():
    print(f"Lietotāji: {EVAL_USERS}  |  Dziesmas: {EVAL_TOP_TRACKS}  |  Izpildes: {EVAL_RUNS}")

    st   = load_state()
    plan = [(uid, tid, lcf, cf_at(uid, tid, st["uf_s"], st["if_s"]))
            for uid in EVAL_USERS
            for tid, lcf in top_n(uid, EVAL_TOP_TRACKS, st["plays"], st["uf_l"], st["if_l"])]

    total = len(plan) * EVAL_RUNS
    lime_rows, shap_rows = [], []

    for i, (uid, tid, lcf, scf) in enumerate(plan):
        for run in range(EVAL_RUNS):
            s1 = lime_config.RANDOM_SEED + run
            s2 = lime_config.RANDOM_SEED + run + 100
            done = i * EVAL_RUNS + run + 1
            print(f"  [{done}/{total}]  user={uid}  track={tid}  run={run}", flush=True)

            try:
                lf, ls, lc, lr = run_lime(uid, tid, lcf, st, s1, s2)
                lime_rows.append({"user_idx": uid, "track_idx": tid, "run": run,
                    "fidelity_score": lf["fidelity_score"], "fidelity_decision": lf["fidelity_decision"],
                    "fidelity_abs_diff": lf["fidelity_abs_diff"], "simplicity_tau_010": ls["simplicity_tau_010"],
                    "simplicity_tau_005": ls["simplicity_tau_005"], "simplicity_tau_001": ls["simplicity_tau_001"],
                    "consistency_score": lc["consistency_score"], "robustness_score": lr["robustness_score"],
                    "robustness_std": lr["robustness_std"]})
            except Exception as e:
                print(f"    LIME kļūda: {e}")

            try:
                sf, ss, sc, sr = run_shap(uid, tid, st, s1, s2)
                shap_rows.append({"user_idx": uid, "track_idx": tid, "run": run,
                    "fidelity_score": sf["fidelity_score"], "fidelity_decision": sf["fidelity_decision"],
                    "fidelity_abs_diff": sf["fidelity_abs_diff"], "simplicity_tau_010": ss["simplicity_tau_010"],
                    "simplicity_tau_005": ss["simplicity_tau_005"], "simplicity_tau_001": ss["simplicity_tau_001"],
                    "consistency_score": sc["consistency_score"], "robustness_score": sr["robustness_score"],
                    "robustness_std": sr["robustness_std"]})
            except Exception as e:
                print(f"    SHAP kļūda: {e}")

    lime_df = pd.DataFrame(lime_rows)
    shap_df = pd.DataFrame(shap_rows)

    def agg(df):
        return df[METRIC_COLS].agg(["mean", "std"]) if not df.empty else None

    la, sa = agg(lime_df), agg(shap_df)
    cmp_rows = []
    for col in METRIC_COLS:
        lm = la.loc["mean", col] if la is not None else float("nan")
        ld = la.loc["std",  col] if la is not None else float("nan")
        sm = sa.loc["mean", col] if sa is not None else float("nan")
        sd = sa.loc["std",  col] if sa is not None else float("nan")
        delta = sm - lm
        cmp_rows.append({"metric": METRIC_LABELS[col],
            "lime_mean": round(lm, 6), "lime_std": round(ld, 6),
            "shap_mean": round(sm, 6), "shap_std": round(sd, 6),
            "delta_shap_minus_lime": round(delta, 6)})

    os.makedirs(OUT_DIR, exist_ok=True)
    lime_df.to_csv(os.path.join(OUT_DIR, "lime_eval_res.csv"),       index=False)
    shap_df.to_csv(os.path.join(OUT_DIR, "shap_eval_res.csv"),       index=False)
    pd.DataFrame(cmp_rows).to_csv(os.path.join(OUT_DIR, "combined_comparison.csv"), index=False)
    print("Pabeigts.")

    return lime_df, shap_df, pd.DataFrame(cmp_rows)

if __name__ == "__main__":
    run_combined_evaluation()