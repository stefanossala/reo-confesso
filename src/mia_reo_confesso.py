"""
MIA sul progetto 'reo-confesso' — aggancio alla pipeline reale.

Replica il preprocessing di src/train_model.py, carica i target .keras reali e
lancia tre attacchi (Threshold / Shadow-Shokri / LiRA-Carlini) contro ciascun
target (normal, overfit). Gli shadow model usano la STESSA config del target
e sono addestrati su data/shadow_pool.csv (disgiunto da members e non-members).
"""
from __future__ import annotations
import argparse, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[1]  # radice del repo
DATA, MODELS = ROOT / "data", ROOT / "models"
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)

SEED = 42
LABEL = "onset_5y"
NUMERIC = ["age", "bmi", "activity_h", "systolic_bp", "glucose"]
BINARY = ["smoker", "family_history"]
ZONES = list("ABCDE")
FEATURE_ORDER = NUMERIC + ["sex_M"] + BINARY + [f"zone_{z}" for z in ZONES]

# Config degli shadow model. Per 'normal' identica al training.
# Per 'overfit' abbassiamo epoche (220) e n. shadow di default: su 1000 righe la
# rete memorizza comunque al 100% (train acc 1.0), quindi il carattere 'overfit'
# e' preservato, ma la run rientra nel limite di tempo del background.
CONFIGS = {
    "normal":  {"hidden": [64, 64],  "l2": 1e-3, "dropout": 0.2, "epochs": 36,  "batch": 32},
    "overfit": {"hidden": [256, 256], "l2": 0.0,  "dropout": 0.0, "epochs": 220, "batch": 32},
}
# numero di shadow per target (gli overfit costano molto piu' tempo)
NSHADOW = {"normal": 32, "overfit": 24}
CELLS = OUT / "cells"; CELLS.mkdir(exist_ok=True)


# --- preprocessing identico al training -----------------------------------
def to_feature_frame(df):
    out = df[NUMERIC + BINARY].astype("float32").copy()
    out["sex_M"] = (df["sex"] == "M").astype("float32")
    for z in ZONES:
        out[f"zone_{z}"] = (df["zone"] == z).astype("float32")
    return out[FEATURE_ORDER]

def fit_scaler(x):
    mean = x.mean(); std = x.std(ddof=0).replace(0, 1.0)
    return {"mean": mean.values, "std": std.values}

def transform(x, sc):
    return ((x.values - sc["mean"]) / sc["std"]).astype("float32")

def build_model(n_features, cfg):
    reg = keras.regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None
    inp = keras.Input(shape=(n_features,))
    x = inp
    for u in cfg["hidden"]:
        x = keras.layers.Dense(u, activation="relu", kernel_regularizer=reg)(x)
        if cfg["dropout"] > 0:
            x = keras.layers.Dropout(cfg["dropout"])(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m


# --- utility di membership --------------------------------------------------
def p_onset(model, X):
    return model.predict(X, verbose=0, batch_size=512).ravel()

def p_true_class(model, X, y):
    """Probabilita' che il modello assegna alla classe VERA di ciascun esempio."""
    p = p_onset(model, X)
    return np.where(y == 1, p, 1.0 - p)

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)


# --- attacchi ---------------------------------------------------------------
def attack_threshold(model, X, y):
    """Punteggio = confidenza sulla classe vera (piu' alta => piu' 'member')."""
    return p_true_class(model, X, y)

def _feat(model, X, y):
    pt = p_true_class(model, X, y)
    return np.column_stack([pt, logit(pt), -np.log(np.clip(pt, 1e-7, 1))])

def train_shadow_attacker(cfg, Xpool_df, ypool, n_shadows, sub, rng):
    feats, labs = [], []
    for s in range(n_shadows):
        idx = rng.permutation(len(Xpool_df))
        in_idx, out_idx = idx[:sub], idx[sub:2 * sub]
        fr_in = Xpool_df.iloc[in_idx]
        sc = fit_scaler(fr_in)
        keras.utils.set_random_seed(1000 + s)
        m = build_model(len(FEATURE_ORDER), cfg)
        m.fit(transform(fr_in, sc), ypool[in_idx], epochs=cfg["epochs"], batch_size=cfg["batch"], verbose=0)
        Xi = transform(fr_in, sc); Xo = transform(Xpool_df.iloc[out_idx], sc)
        feats.append(_feat(m, Xi, ypool[in_idx]));  labs.append(np.ones(len(in_idx)))
        feats.append(_feat(m, Xo, ypool[out_idx])); labs.append(np.zeros(len(out_idx)))
        keras.backend.clear_session()
    atk = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    atk.fit(np.vstack(feats), np.concatenate(labs))
    return atk

def attack_shadow(atk, model, X, y):
    return atk.predict_proba(_feat(model, X, y))[:, 1]

def run_lira_offline(name, cfg, eval_df, y_eval, Xpool_df, ypool, n_shadows, sub, rng):
    """LiRA offline: shadow addestrati solo 'OUT' rispetto agli eval point.
    Per ogni eval point stima la gaussiana di phi=logit(p_true) quando NON e'
    nel training. La matrice phi viene messa in cache: il ri-scoring e' istantaneo.
    """
    cache = CELLS / f"lira_phi_{name}.npy"
    if cache.exists():
        phis = np.load(cache)
        print(f"     (phi shadow da cache: {phis.shape})", flush=True)
    else:
        Xeval_raw = to_feature_frame(eval_df)
        phis = np.zeros((n_shadows, len(eval_df)), dtype=np.float64)
        for s in range(n_shadows):
            idx = rng.permutation(len(Xpool_df))[:sub]
            fr_in = Xpool_df.iloc[idx]
            sc = fit_scaler(fr_in)
            keras.utils.set_random_seed(5000 + s)
            m = build_model(len(FEATURE_ORDER), cfg)
            m.fit(transform(fr_in, sc), ypool[idx], epochs=cfg["epochs"], batch_size=cfg["batch"], verbose=0)
            phis[s] = logit(p_true_class(m, transform(Xeval_raw, sc), y_eval))
            keras.backend.clear_session()
        np.save(cache, phis)
    mu_out = phis.mean(0)
    s_out = phis.std(0)
    s_out = np.maximum(s_out, 0.25 * float(np.mean(s_out)))  # floor per stabilita'
    return mu_out, s_out

def lira_scores(model, eval_df, y_eval, mu_out, s_out, scaler_target):
    """Z-score standardizzato: (phi_obs - mu_out)/s_out. Piu' alto => piu' 'member'
    (l'osservazione sta nella coda destra della distribuzione OUT dell'esempio)."""
    Xeval = transform(to_feature_frame(eval_df), scaler_target)
    phi_obs = logit(p_true_class(model, Xeval, y_eval))
    return (phi_obs - mu_out) / s_out


# --- valutazione ------------------------------------------------------------
def tpr_at(fpr, tpr, t):
    i = max(np.searchsorted(fpr, t, side="right") - 1, 0)
    return float(tpr[i])

def evaluate(name, y_member, scores):
    fpr, tpr, _ = roc_curve(y_member, scores)
    a = float(auc(fpr, tpr))
    res = {"auc": round(a, 4),
           "tpr@0.1%": round(tpr_at(fpr, tpr, 0.001), 4),
           "tpr@1%": round(tpr_at(fpr, tpr, 0.01), 4),
           "tpr@10%": round(tpr_at(fpr, tpr, 0.10), 4)}
    print(f"   [{name}] AUC={res['auc']}  TPR@0.1%={res['tpr@0.1%']}  "
          f"TPR@1%={res['tpr@1%']}  TPR@10%={res['tpr@10%']}")
    return fpr.tolist(), tpr.tolist(), res


def save_cell(target, attack, fpr, tpr, res):
    (CELLS / f"{target}__{attack}.json").write_text(
        json.dumps({"res": res, "fpr": fpr, "tpr": tpr}))

def have_cell(target, attack):
    return (CELLS / f"{target}__{attack}.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["normal", "overfit"])
    ap.add_argument("--attacks", nargs="+", default=["Threshold", "Shadow", "LiRA"])
    ap.add_argument("--sub", type=int, default=1000, help="dim. training di ogni shadow")
    ap.add_argument("--assemble", action="store_true", help="solo: assembla report+plot dalle celle")
    args = ap.parse_args()

    train_df = pd.read_csv(DATA / "target_train.csv")   # members
    out_df   = pd.read_csv(DATA / "target_out.csv")     # non-members
    pool_df  = pd.read_csv(DATA / "shadow_pool.csv")    # mondo dell'attaccante
    Xpool_df = to_feature_frame(pool_df); ypool = pool_df[LABEL].values
    y_members, y_nonmembers = train_df[LABEL].values, out_df[LABEL].values
    eval_df = pd.concat([train_df, out_df], ignore_index=True)
    y_eval = eval_df[LABEL].values
    y_is_member = np.concatenate([np.ones(len(train_df)), np.zeros(len(out_df))])
    sc_t = fit_scaler(to_feature_frame(train_df))
    Xeval_t = transform(to_feature_frame(eval_df), sc_t)

    if not args.assemble:
        for name in args.targets:
            cfg, ns = CONFIGS[name], NSHADOW[name]
            print(f"\n================  TARGET: {name}  (shadow={ns})  ================", flush=True)
            model = keras.models.load_model(MODELS / f"target_{name}.keras")
            pm, po = p_onset(model, transform(to_feature_frame(train_df), sc_t)), p_onset(model, transform(to_feature_frame(out_df), sc_t))
            chk = {"members": {"accuracy": round(accuracy_score(y_members, pm >= .5), 4),
                               "auc": round(roc_auc_score(y_members, pm), 4),
                               "loss": round(log_loss(y_members, np.clip(pm, 1e-7, 1-1e-7)), 4)},
                   "non_members": {"accuracy": round(accuracy_score(y_nonmembers, po >= .5), 4),
                                   "auc": round(roc_auc_score(y_nonmembers, po), 4),
                                   "loss": round(log_loss(y_nonmembers, np.clip(po, 1e-7, 1-1e-7)), 4)}}
            (CELLS / f"{name}__metrics.json").write_text(json.dumps(chk))
            print(f"  verifica metriche (vs metrics.json): {chk}", flush=True)

            if "Threshold" in args.attacks and not have_cell(name, "Threshold"):
                print("  -> Threshold (Yeom)...", flush=True)
                save_cell(name, "Threshold", *evaluate("Threshold", y_is_member, attack_threshold(model, Xeval_t, y_eval)))

            if "Shadow" in args.attacks and not have_cell(name, "Shadow"):
                print(f"  -> Shadow (Shokri) — {ns} shadow...", flush=True)
                atk = train_shadow_attacker(cfg, Xpool_df, ypool, ns, args.sub, np.random.default_rng(SEED))
                save_cell(name, "Shadow", *evaluate("Shadow", y_is_member, attack_shadow(atk, model, Xeval_t, y_eval)))

            if "LiRA" in args.attacks and not have_cell(name, "LiRA"):
                print(f"  -> LiRA offline (Carlini) — {ns} shadow...", flush=True)
                mu_out, s_out = run_lira_offline(name, cfg, eval_df, y_eval, Xpool_df, ypool, ns, args.sub, np.random.default_rng(SEED))
                save_cell(name, "LiRA", *evaluate("LiRA", y_is_member, lira_scores(model, eval_df, y_eval, mu_out, s_out, sc_t)))

    # assembla sempre quello che c'e'
    report = {"config": {"sub": args.sub, "n_shadows": NSHADOW}, "targets": {}}
    for name in ["normal", "overfit"]:
        mpath = CELLS / f"{name}__metrics.json"
        entry = {"metrics_check": json.loads(mpath.read_text()) if mpath.exists() else None,
                 "attacks": {}, "roc": {}}
        for atk in ["Threshold", "Shadow", "LiRA"]:
            cp = CELLS / f"{name}__{atk}.json"
            if cp.exists():
                d = json.loads(cp.read_text())
                entry["attacks"][atk] = d["res"]; entry["roc"][atk] = {"fpr": d["fpr"], "tpr": d["tpr"]}
        report["targets"][name] = entry
    (OUT / "attack_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport assemblato in {OUT/'attack_report.json'}", flush=True)


if __name__ == "__main__":
    main()
