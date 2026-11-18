"""PathSnap API (runs on Laptop 1).

Start:  uvicorn api.main:app --host 0.0.0.0 --port 8000
  GET  /health            -> status
  GET  /trips             -> trip ids available for evaluation
  POST /predict {trip_id} -> GPS points, predicted route geometry, server-side metrics

Trips are read from data/evaluator_test/ (drop the organisers' trip folders there).
If that folder is missing/empty we fall back to data/student_test/ so the demo can be rehearsed.
"""
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.evaluate import evaluate  # noqa: E402
from src.pipeline import Predictor, route_polylines  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")


def trip_root() -> Path:
    eval_dir = Path(os.environ.get("PATHSNAP_TRIPS", ROOT / "data" / "evaluator_test"))
    if eval_dir.exists() and any(p.is_dir() for p in eval_dir.iterdir()):
        return eval_dir
    return ROOT / "data" / "student_test"


app = FastAPI(title="PathSnap API")
predictor = Predictor()
log.info("Model loaded. Serving trips from %s", trip_root())


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "trip_source": trip_root().name}


@app.get("/trips")
def trips():
    root = trip_root()
    ids = sorted(p.name for p in root.iterdir() if p.is_dir())
    log.info("GET /trips -> %d trips from %s", len(ids), root.name)
    return {"source": root.name, "trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    trip_dir = trip_root() / req.trip_id
    if not trip_dir.is_dir():
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    log.info("POST /predict trip=%s ...", req.trip_id)
    res = predictor.predict(trip_dir)
    metrics = evaluate(trip_dir, res)
    track = res["trip"]["track"]
    step = max(1, len(track) // 3000)  # keep the response light for the map
    log.info("  trip=%s points=%d candidates=%d time=%sms metrics=%s", req.trip_id, len(res["clean"]),
             res["n_candidates_rows"], res["time_ms"],
             {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})
    return {
        "trip_id": req.trip_id,
        "n_gps_points": len(track),
        "n_matched_points": int((res["pred_arcs"] >= 0).sum()),
        "time_ms": res["time_ms"],
        "gps": track[["lat", "lon"]].to_numpy()[::step].tolist(),
        "pred_route": route_polylines(res["pred_arcs"], res["trip"]["nodes"], res["trip"]["arcs"]),
        "metrics": metrics,
    }
