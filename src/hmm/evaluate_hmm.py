"""Classifier-only vs classifier + HMM/Viterbi on the student_test trips.

Beta (transition scale, metres) is chosen on student_test, which the guidelines allow
("optimise decoder parameters using training and student-testing data only").

Run: python -m src.hmm.evaluate_hmm
Outputs: experiments/hmm_beta_search.csv, experiments/hmm_comparison.csv, artifacts/hmm.json,
         reports/figures/hmm_comparison.png
"""
import json
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.geo import project, trip_epsg
from src.data.loader import list_trips, load_trip
from src.features.candidates import FEATURES
from src.hmm.viterbi import RoadGraph, decode
from src.models import metrics as M
from src.models.train_compare import EXP, FIG, ROOT

BETAS = [2, 5, 10, 20, 50, 100]


def prepare(model, partition="student_test"):
    feats = pd.read_parquet(EXP / f"features_{partition}.parquet")
    pts = pd.read_parquet(EXP / f"points_{partition}.parquet")
    out = []
    for tid in list_trips(partition):
        trip = load_trip(partition, tid)
        c = feats[feats.trip == tid].reset_index(drop=True)
        clean = pts[pts.trip == tid].reset_index(drop=True)
        xy = np.column_stack(project(trip["nodes"].lon, trip["nodes"].lat, trip_epsg(trip["track"])))
        proba = model.predict_proba(c[FEATURES].to_numpy())[:, 1]
        best = M.top1_predictions(c, proba)
        clf = np.full(len(clean), -1)
        clf[best.point_idx.to_numpy()] = best.pred_arc.to_numpy()
        out.append(dict(tid=tid, trip=trip, c=c, clean=clean, proba=proba, clf=clf, graph=RoadGraph(xy, trip["arcs"], clean)))
    return out


def score(items, preds):
    acc = [np.mean(p == it["clean"].true_arc.to_numpy()) for it, p in zip(items, preds)]
    cont = [M.route_continuity(p, it["clean"].seg_id, it["trip"]["arcs"]) for it, p in zip(items, preds)]
    n = [len(it["clean"]) for it in items]
    return float(np.average(acc, weights=n)), float(np.mean(cont)), acc, cont


def main(model_path=ROOT / "artifacts" / "model.joblib"):
    model = joblib.load(model_path)
    items = prepare(model)
    base_acc, base_cont, b_acc, b_cont = score(items, [it["clf"] for it in items])
    print(f"classifier only: top1 {base_acc:.4f} continuity {base_cont:.4f}")

    search = []
    for beta in BETAS:
        t0 = time.time()
        preds = [decode(it["c"], it["proba"], it["clean"], it["graph"], beta=beta) for it in items]
        acc, cont, _, _ = score(items, preds)
        search.append({"beta": beta, "top1_acc": acc, "route_continuity": cont, "seconds": time.time() - t0})
        print(search[-1])
    s = pd.DataFrame(search); s.to_csv(EXP / "hmm_beta_search.csv", index=False)
    best_beta = float(s.loc[s.top1_acc.idxmax(), "beta"])

    preds = [decode(it["c"], it["proba"], it["clean"], it["graph"], beta=best_beta) for it in items]
    h_acc, h_cont, h_accs, h_conts = score(items, preds)
    per_trip = pd.DataFrame({"trip": [it["tid"] for it in items], "points": [len(it["clean"]) for it in items],
                             "clf_top1": b_acc, "hmm_top1": h_accs, "clf_continuity": b_cont, "hmm_continuity": h_conts})
    per_trip.loc[len(per_trip)] = ["ALL (point-weighted acc)", per_trip.points.sum(), base_acc, h_acc, base_cont, h_cont]
    per_trip.to_csv(EXP / "hmm_comparison.csv", index=False)
    print(per_trip.round(4).to_string(index=False))
    (ROOT / "artifacts" / "hmm.json").write_text(json.dumps({"beta": best_beta, "max_route_m": 2000, "local_buffer_m": 300}, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(per_trip) - 1); w = 0.4
    for a, k, title in [(ax[0], "top1", "Top-1 accuracy"), (ax[1], "continuity", "Route continuity")]:
        a.bar(x - w / 2, per_trip[f"clf_{k}"][:-1], w, label="classifier only")
        a.bar(x + w / 2, per_trip[f"hmm_{k}"][:-1], w, label=f"+ HMM (beta={best_beta:g})")
        a.set_xticks(x, per_trip.trip[:-1], rotation=45); a.set_title(f"{title} per student_test trip"); a.legend()
        a.set_ylim(0.5, 1.0)
    plt.tight_layout(); plt.savefig(FIG / "hmm_comparison.png", dpi=120); plt.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
