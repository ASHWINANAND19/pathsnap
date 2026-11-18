"""Final Folium maps: raw GPS, nearby road network, reference route, predicted route (classifier + HMM).

Run: python -m src.viz.final_maps
Output: maps/final_<trip>.html for a few student_test trips (a good, an average and a weak one).
"""
from pathlib import Path

from src.data.loader import DATA_DIR, load_trip
from src.pipeline import Predictor
from src.viz.maps import trip_map

ROOT = Path(__file__).resolve().parents[2]
TRIPS = ["00000000", "00000076", "00000077"]  # best-ish, mid, weakest student_test trips


def main():
    pred = Predictor()
    for tid in TRIPS:
        res = pred.predict(DATA_DIR / "student_test" / tid)
        m = trip_map(load_trip("student_test", tid), pred_arcs=res["pred_arcs"], gps_every=2)
        out = ROOT / "maps" / f"final_{tid}.html"
        m.save(out)
        print("saved", out)


if __name__ == "__main__":
    main()
