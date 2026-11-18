# PathSnap: design

## Intent
Build an end-to-end map-matcher for the PES ML hackathon. Every choice should be simple enough for any team member to defend in the viva.
Success looks like this:
- it works for all 10 evaluator trips over two laptops
- the report is backed by real runs
- the code is clear

## Goals / non-goals / anti-goals
- **Goals:** EDA with Folium, a candidate classifier with at least 5 models, Optuna tuning, HMM/Viterbi, FastAPI + Streamlit, and a report.
- **Non-goals:** state-of-the-art accuracy, deep sequence models, real-time streaming.
- **Anti-goals (these would mean failure):**
  - evaluator ground truth reaching the UI or the model
  - hardcoded outputs
  - fabricated numbers
  - code nobody can explain

## Constraints (from the guidelines)
- Train on the 80 train trips. The 10 student-test trips are for comparison only. Evaluator trips are for the demo only.
- At least 5 models, using group-aware CV (GroupKFold by trip).
- Bayesian HPO with the trial log retained.
- Two laptops: the API on L1, the UI on L2. The API computes accuracy server-side and logs requests in the terminal.
- A public GitHub repo and a PDF report. Deadlines: first evaluation at 12:00, submission at 16:00.

## Approaches considered (labelling)
- **A. Nearest route arc (CHOSEN by the user).** A point's label is the route arc with the smallest perpendicular distance to it. It is simple. It can mislabel points at junctions and on backtracking routes; we report that as a limitation.
- **B. Order-aware monotone alignment.** It is more accurate on loops, but harder to explain and takes longer. Rejected for time.

## Design
- **Projection:** UTM zone 32N, EPSG:32632, so all distances are in metres.
- **Cleaning:**
  - Drop a point if it is less than 3 m from the last kept point.
  - Split a trip into segments wherever the gap is over 60 s.
- **Candidates:** arcs within R = 50 m of the point, at most K = 8, found with a shapely STRtree. A point that gets no candidates is marked "no candidate".
- **Label:** a candidate is 1 if it is the route arc nearest to the point (option A), otherwise 0. If that arc is not among the candidates, it counts as a candidate miss.
- **Features:**
  - dist_m
  - dist_rank
  - dist_minus_min
  - heading_diff_deg (GPS movement bearing compared with the arc bearing u→v)
  - arc_len_m
  - speed_mps
  - n_candidates
  - frac_along
  - is_closest
- **Models (binary, scored per candidate):**
  - LogisticRegression
  - KNN
  - RandomForest
  - LightGBM
  - MLP
  - GaussianNB as an optional sixth
- **Metrics:**
  - CV macro F1 (GroupKFold, 5 folds)
  - Student-test: Top-1 accuracy per point, macro F1, log loss, candidate miss rate, route continuity, inference time
- **HPO:** Optuna TPE on LightGBM, 30 trials. The objective is mean CV macro F1. Trials are saved to CSV.
- **HMM/Viterbi:**
  - States: the candidate arcs at each point.
  - Emission: log of the classifier probability.
  - Transition: exp(-|network distance − GPS distance| / β), where network distance comes from a Dijkstra search on the directed arc graph, capped at 2 km. A cost of 0 applies when staying on the same arc.
  - Chains break at gaps or where no candidates exist.
  - β is tuned on the train and student-test trips.
- **API (FastAPI):**
  - `GET /trips` lists trips from `data/evaluator_test/`, falling back to `data/student_test/`.
  - `POST /predict {trip_id}` returns the GPS points, the predicted arc geometry and the metrics. The ground truth stays server-side.
  - Requests are logged to the console.
- **UI (Streamlit):** in a self-contained `ui/` folder with its own requirements. It has an API URL input, a trip dropdown, a Folium map and metric cards.
- **Viva support:** heavily commented code, plus docs/VIVA_NOTES.md.
