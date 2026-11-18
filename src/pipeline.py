"""End-to-end inference for ONE trip: raw files -> predicted arc per GPS point.

Deliberately reads only .track, .nodes and .arcs. The .route (ground truth) is never
touched here, so it cannot leak into the prediction; api/evaluate.py scores the result afterwards.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.geo import project, trip_epsg
from src.features.candidates import FEATURES, build_candidates
from src.hmm.viterbi import RoadGraph, decode

ROOT = Path(__file__).resolve().parents[1]


def read_trip_inputs(trip_dir: Path) -> dict:
    tid = trip_dir.name
    read = lambda ext, cols: pd.read_csv(trip_dir / f"{tid}.{ext}", sep=r"\s+", header=None, names=cols)
    return {"track": read("track", ["lon", "lat", "t"]), "nodes": read("nodes", ["lon", "lat"]),
            "arcs": read("arcs", ["u", "v"]), "route": None}


class Predictor:
    def __init__(self, artifacts=ROOT / "artifacts"):
        self.model = joblib.load(artifacts / "model.joblib")
        self.beta = json.loads((artifacts / "hmm.json").read_text())["beta"]

    def predict(self, trip_dir: Path) -> dict:
        t0 = time.time()
        trip = read_trip_inputs(trip_dir)
        clean, cands = build_candidates(trip, with_labels=False)
        proba = self.model.predict_proba(cands[FEATURES].to_numpy())[:, 1] if len(cands) else np.array([])
        epsg = trip_epsg(trip["track"])
        xy = np.column_stack(project(trip["nodes"].lon, trip["nodes"].lat, epsg))
        pred = decode(cands, proba, clean, RoadGraph(xy, trip["arcs"], clean), beta=self.beta)
        return {"trip": trip, "clean": clean, "pred_arcs": pred, "n_candidates_rows": len(cands),
                "time_ms": round(1000 * (time.time() - t0), 1)}


def route_polylines(pred_arcs, nodes, arcs):
    """Predicted arcs (in driving order, consecutive duplicates removed) as [[lat, lon], [lat, lon]] pairs."""
    a = np.asarray(pred_arcs)
    a = a[a >= 0]
    if len(a) == 0:
        return []
    a = a[np.r_[True, a[1:] != a[:-1]]]
    u, v = arcs.u.to_numpy()[a], arcs.v.to_numpy()[a]
    P = nodes[["lat", "lon"]].to_numpy()
    return [[P[i].tolist(), P[j].tolist()] for i, j in zip(u, v)]
