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

## Reproduce everything (Laptop 1)

```bash
python -m src.data.inventory            # dataset inventory
python -m src.features.build_dataset    # candidates + labels + features -> experiments/*.parquet
python -m src.models.train_compare      # 6 models, GroupKFold CV + student_test -> model_comparison.csv
python -m src.models.tune 25            # Optuna (Bayesian) tuning of the MLP -> artifacts/model.joblib
python -m src.hmm.evaluate_hmm          # classifier vs classifier+HMM, picks beta -> artifacts/hmm.json
python -m pytest -q tests               # unit tests (labels, gaps, Viterbi)
```
EDA: `notebooks/01_eda.ipynb`. Figures go to `reports/figures/`, maps to `maps/`.

## Running the demo

**Laptop 1 (API):**
```bash
# put the evaluator trip folders in data/evaluator_test/<trip_id>/ (same file format)
uvicorn api.main:app --host 0.0.0.0 --port 8000
ipconfig getifaddr en0                  # this laptop's IP, to give to Laptop 2
```
If `data/evaluator_test/` is empty, the API serves `data/student_test/` instead, which is useful for rehearsal.

**Laptop 2 (UI):** see [ui/README.md](ui/README.md). In short, `pip install -r ui/requirements-ui.txt && streamlit run ui/app.py`, then enter `http://<Laptop-1-IP>:8000`.

Ground truth (`.route`) is read only by `api/evaluate.py` on Laptop 1, after prediction.
