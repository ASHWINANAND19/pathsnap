import numpy as np
import pandas as pd

from src.data.geo import clean_track, project
from src.hmm.viterbi import RoadGraph, decode

EPSG = 32632


def _setup():
    """Road A: arcs 0 -> 1 connected end-to-end (west->east). Arc 2 is a separate parallel road."""
    nodes = pd.DataFrame({"lon": [9.000, 9.002, 9.004, 9.002, 9.004],
                          "lat": [0.0, 0.0, 0.0, 0.0003, 0.0003]})
    arcs = pd.DataFrame({"u": [0, 1, 3], "v": [1, 2, 4]})
    track = pd.DataFrame({"lon": [9.0015, 9.0025], "lat": [0.00012, 0.00012], "t": [0.0, 10.0]})
    clean = clean_track(track, EPSG)
    xy = np.column_stack(project(nodes.lon, nodes.lat, EPSG))
    return clean, RoadGraph(xy, arcs, clean)


def test_viterbi_prefers_connected():
    clean, g = _setup()
    # point 0 is surely on arc 0; at point 1 the classifier slightly prefers disconnected arc 2
    cands = pd.DataFrame({"point_idx": [0, 1, 1], "arc_id": [0, 1, 2], "frac_along": [0.75, 0.25, 0.25]})
    proba = np.array([0.9, 0.45, 0.55])
    assert decode(cands, proba, clean, g, beta=20).tolist() == [0, 1]


def test_viterbi_no_candidates():
    clean, g = _setup()
    cands = pd.DataFrame({"point_idx": [1], "arc_id": [1], "frac_along": [0.25]})
    assert decode(cands, np.array([0.8]), clean, g).tolist() == [-1, 1]
