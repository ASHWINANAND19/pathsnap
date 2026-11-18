"""HMM + Viterbi route reconstruction.

HMM view of map matching:
  * Hidden state at time i  = the road arc the car is really on (one of point i's candidates).
  * Observation at time i   = the GPS point.
  * Emission  P(GPS_i | arc) = classifier probability for that (point, arc) row.
  * Transition P(arc_b at i+1 | arc_a at i) = exp(-|d_route - d_gps| / beta)
        d_gps   = straight-line distance between the two GPS points
        d_route = driving distance along the road network from the projection on arc_a
                  to the projection on arc_b (same arc: distance along it; otherwise
                  rest of arc_a + shortest path v_a -> u_b + start of arc_b).
    If the car could not drive from a to b (no path within MAX_ROUTE_M) the transition is
    impossible. A realistic move has d_route ~= d_gps, so the penalty is ~0.
  * Viterbi = dynamic programming that finds the single most probable arc sequence:
        score_i(b) = log e_i(b) + max_a [ score_{i-1}(a) + log T(a -> b) ]
    and back-tracks the argmax pointers.
  * Breaks: at GPS gaps (> 60 s, seg_id changes), at points with no candidates, and when no
    candidate is reachable from any previous one, we restart the chain (no transition term).
"""
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

MAX_ROUTE_M = 2000.0   # Dijkstra search limit
LOCAL_BUFFER_M = 300.0  # only roads this close to the trip are used for shortest paths
BACKTRACK_TOL_M = 5.0   # allow GPS jitter to move slightly "backwards" on the same arc
EPS = 1e-6


class RoadGraph:
    """Directed graph of arcs near one trip, with node-to-node shortest path lookups."""

    def __init__(self, nodes_xy, arcs, clean):
        lines = shapely.linestrings(np.stack([nodes_xy[arcs.u.to_numpy()], nodes_xy[arcs.v.to_numpy()]], axis=1))
        pts = shapely.points(clean.x.to_numpy(), clean.y.to_numpy())
        near = np.unique(shapely.STRtree(lines).query(pts, predicate="dwithin", distance=LOCAL_BUFFER_M)[1])
        self.arc_len = shapely.length(lines)
        u, v = arcs.u.to_numpy()[near], arcs.v.to_numpy()[near]
        self.local_nodes, inv = np.unique(np.concatenate([u, v]), return_inverse=True)
        self.node_to_local = {n: i for i, n in enumerate(self.local_nodes)}
        lu, lv = inv[: len(u)], inv[len(u):]
        n = len(self.local_nodes)
        self.csr = csr_matrix((self.arc_len[near] + EPS, (lu, lv)), shape=(n, n))
        self.arcs_u, self.arcs_v = arcs.u.to_numpy(), arcs.v.to_numpy()
        self._cache = {}

    def sp_from(self, node):
        """Distances from one global node id to all local nodes (cached)."""
        if node not in self._cache:
            li = self.node_to_local.get(node)
            self._cache[node] = (np.full(len(self.local_nodes), np.inf) if li is None
                                 else dijkstra(self.csr, indices=li, limit=MAX_ROUTE_M))
        return self._cache[node]

    def route_dist(self, a_arcs, a_frac, b_arcs, b_frac):
        """Matrix [len(a), len(b)] of driving distances between projections."""
        D = np.full((len(a_arcs), len(b_arcs)), np.inf)
        la, lb = self.arc_len[a_arcs], self.arc_len[b_arcs]
        for i, (aa, fa) in enumerate(zip(a_arcs, a_frac)):
            sp = self.sp_from(self.arcs_v[aa])
            ub_local = np.array([self.node_to_local.get(n, -1) for n in self.arcs_u[b_arcs]])
            via = np.where(ub_local >= 0, sp[np.maximum(ub_local, 0)], np.inf)
            D[i] = (1 - fa) * la[i] + via + b_frac * lb
            same = b_arcs == aa
            along = (b_frac - fa) * la[i]
            D[i, same] = np.where(along >= -BACKTRACK_TOL_M, np.abs(along), np.inf)[same]
        return D


def decode(cands: pd.DataFrame, proba: np.ndarray, clean: pd.DataFrame, graph: RoadGraph, beta=20.0):
    """Return predicted arc id per cleaned point (-1 = no candidate).

    cands: rows for ONE trip with point_idx, arc_id, frac_along; proba aligned to cands rows.
    """
    n = len(clean)
    pred = np.full(n, -1, dtype=int)
    c = cands[["point_idx", "arc_id", "frac_along"]].copy()
    c["logp"] = np.log(np.clip(proba, EPS, 1.0))
    groups = {p: g for p, g in c.groupby("point_idx")}
    xy, seg = clean[["x", "y"]].to_numpy(), clean.seg_id.to_numpy()

    # Viterbi state for the current chain
    chain = []  # list of (point_idx, arcs, backpointers)
    score = None
    prev = None

    def flush():
        """Back-track the finished chain and write predictions."""
        if not chain:
            return
        j = int(np.argmax(score))
        for p, arcs, back in reversed(chain):
            pred[p] = arcs[j]
            j = back[j] if back is not None else j
        chain.clear()

    for i in range(n):
        g = groups.get(i)
        if g is None:                       # no candidates -> break the chain
            flush(); score = None; prev = None
            continue
        arcs, frac, logp = g.arc_id.to_numpy(), g.frac_along.to_numpy(), g.logp.to_numpy()
        if prev is None or seg[i] != seg[prev[0]]:
            flush(); score = logp.copy(); chain.append((i, arcs, None)); prev = (i, arcs, frac)
            continue
        d_gps = np.hypot(*(xy[i] - xy[prev[0]]))
        D = graph.route_dist(prev[1], prev[2], arcs, frac)
        logT = -np.abs(D - d_gps) / beta                # -inf where unreachable
        total = score[:, None] + logT
        back = np.argmax(total, axis=0)
        best = total[back, np.arange(len(arcs))]
        if not np.isfinite(best).any():             # nothing reachable -> restart chain
            flush(); score = logp.copy(); chain.append((i, arcs, None))
        else:
            score = best + logp
            chain.append((i, arcs, back))
        prev = (i, arcs, frac)
    flush()
    return pred
