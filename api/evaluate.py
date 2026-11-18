"""Server-side scoring. The ONLY place in the API that reads a .route file.

Ground truth stays on Laptop 1: only the resulting numbers are sent to the UI.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.geo import project, trip_epsg
from src.features.candidates import true_arcs
from src.models.metrics import route_continuity


def evaluate(trip_dir: Path, result: dict) -> dict:
    trip, clean, pred = result["trip"], result["clean"], np.asarray(result["pred_arcs"])
    cont = route_continuity(pred, clean.seg_id, trip["arcs"])
    route_file = trip_dir / f"{trip_dir.name}.route"
    out = {"route_continuity": cont, "has_reference": route_file.exists()}
    if not route_file.exists():
        return out | {"top1_accuracy": None, "route_arc_precision": None, "route_arc_recall": None}
    route = pd.read_csv(route_file, header=None, names=["arc_id"])
    nodes_xy = np.column_stack(project(trip["nodes"].lon, trip["nodes"].lat, trip_epsg(trip["track"])))
    t_arc, _ = true_arcs(clean, nodes_xy, trip["arcs"], route)
    ref, got = set(route.arc_id), set(pred[pred >= 0])
    return out | {
        # observation level: share of GPS points matched to their (option-A) reference arc
        "top1_accuracy": float(np.mean(pred == t_arc)),
        # sequence level: overlap between the predicted set of arcs and the reference route
        "route_arc_precision": len(ref & got) / max(len(got), 1),
        "route_arc_recall": len(ref & got) / max(len(ref), 1),
    }
