"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model (MLP).

Why the MLP? In the 6-model comparison it had the best CV macro F1 (primary metric) and by far
the best log loss, i.e. the best-calibrated probabilities, which matter because the
HMM uses them directly as emission probabilities. Top-1 accuracy was within 0.5 points of RF/LightGBM.

Why Bayesian? Grid search wastes trials on bad regions. TPE builds a probabilistic model of
"which hyperparameters gave good scores" from past trials and samples the next trial where
improvement is most likely, so 30 trials go much further than a 30-point grid.

Objective: mean macro F1 over 5-fold GroupKFold (groups = trips) on the 80 train trips.
Student-test trips are NOT used inside the search — only to compare tuned vs untuned afterwards.

Run: python -m src.models.tune [n_trials]
Outputs: experiments/optuna/trials.csv, experiments/tuned_vs_untuned.csv,
         reports/figures/optuna_history.png, artifacts/model.joblib, artifacts/config.json
"""
import json
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from src.features.candidates import FEATURES, K, R_M
from src.models.train_compare import EXP, FIG, ROOT, SEED, evaluate_student_test, load_train, make_models

LAYERS = {"32": (32,), "64": (64,), "64-32": (64, 32), "128-64": (128, 64), "128-64-32": (128, 64, 32)}
SEARCH_SPACE = {
    "hidden_layers": ("cat", list(LAYERS)),        # network width/depth
    "alpha": ("float_log", 1e-5, 1e-1),            # L2 regularisation strength
    "learning_rate_init": ("float_log", 1e-4, 1e-2),
    "batch_size": ("cat", [128, 256, 512, 1024]),
}


def suggest(trial):
    p = {}
    for name, (kind, lo, *rest) in SEARCH_SPACE.items():
        hi = rest[0] if rest else None
        if kind == "cat":
            p[name] = trial.suggest_categorical(name, lo)
        elif kind == "int":
            p[name] = trial.suggest_int(name, lo, hi)
        elif kind == "float":
            p[name] = trial.suggest_float(name, lo, hi)
        else:
            p[name] = trial.suggest_float(name, lo, hi, log=True)
    return p


def build(params):
    p = dict(params)
    layers = LAYERS[p.pop("hidden_layers")]
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=layers, **p, early_stopping=True,
                                                         max_iter=60, random_state=SEED))


def cv_f1(model, X, y, g):
    scores = []
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        model.fit(X[tr], y[tr])
        scores.append(f1_score(y[va], model.predict(X[va]), average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def main(n_trials=30):
    train = load_train()
    X, y, g = train[FEATURES].to_numpy(), train.label.to_numpy(), train.trip.to_numpy()

    def objective(trial):
        mean, std = cv_f1(build(suggest(trial)), X, y, g)
        trial.set_user_attr("cv_std", std)
        return mean

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    search_s = time.time() - t0
    (EXP / "optuna").mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(EXP / "optuna" / "trials.csv", index=False)
    print(f"best CV macro F1 {study.best_value:.4f} with {study.best_params}  ({search_s:.0f}s)")

    vals = [t.value for t in study.trials]
    plt.figure(figsize=(6, 3.5))
    plt.plot(vals, "o", alpha=0.5, label="trial"); plt.plot(np.maximum.accumulate(vals), "r-", label="best so far")
    plt.xlabel("trial"); plt.ylabel("CV macro F1"); plt.title("Optuna TPE optimisation history"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "optuna_history.png", dpi=120); plt.close()

    # tuned vs untuned, identical protocol
    rows = []
    for label, model in [("untuned", make_models()["mlp"][1]), ("tuned", build(study.best_params))]:
        cv_m, cv_s = cv_f1(model, X, y, g)
        t1 = time.time(); model.fit(X, y); fit_s = time.time() - t1
        res, _ = evaluate_student_test(model, f"mlp_{label}")
        rows.append({"model": f"mlp_{label}", "cv_macro_f1": cv_m, "cv_std": cv_s, "train_s": fit_s,
                     **{f"st_{k}": v for k, v in res.items()}})
        final = model
    cmp = pd.DataFrame(rows)
    cmp.to_csv(EXP / "tuned_vs_untuned.csv", index=False)
    print(cmp.round(4).T)

    (ROOT / "artifacts").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "artifacts" / "model.joblib")
    (ROOT / "artifacts" / "config.json").write_text(json.dumps({
        "model": "MLP (Optuna-tuned)", "features": FEATURES, "best_params": study.best_params,
        "candidate_radius_m": R_M, "max_candidates": K, "n_trials": n_trials,
        "search_seconds": round(search_s, 1), "seed": SEED, "search_space": SEARCH_SPACE,
    }, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
