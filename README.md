# PathSnap — Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon (PES University, 18 Nov 2026).
Match noisy GPS traces to a road network: per-point candidate classifier → HMM/Viterbi → FastAPI (Laptop 1) + UI (Laptop 2).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python -m src.data.inventory   # dataset sanity check -> experiments/dataset_inventory.csv
```

Jupyter kernel: **PathSnap (py3.12)**.

## Layout

| Folder | Contents |
|---|---|
| `data/train` (80 trips), `data/student_test` (10 trips) | Raw `.track/.nodes/.arcs/.route` per trip |
| `src/data` | Loading, inventory, cleaning |
| `src/features` | Candidate generation + feature engineering |
| `src/models` | Training/comparison of ≥5 models, Optuna Bayesian search |
| `src/hmm` | HMM + Viterbi decoder |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Evaluator UI (Laptop 2) |
| `notebooks/` | EDA |
| `experiments/` | Logs, CSV results, Optuna studies |
| `artifacts/` | Saved model + configs |
| `maps/`, `reports/figures/` | Folium HTML maps, plots |
| `docs/guidelines/` | Official hackathon PDFs |

## Running the demo
_TODO: API + UI commands once implemented._
