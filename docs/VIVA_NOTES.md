# PathSnap: viva notes

Every number here comes from a file in `experiments/`. Quote the file if you're asked "where is that from?".

---

## 1. Dataset structure
- **Partitions:** 80 train trips, 10 student_test trips. The 10 evaluator trips are used only at the demo.
- **Files in each trip folder:**
  - `.track`: lon, lat, t in seconds. About 1 Hz (96.5% of steps are exactly 1 s).
  - `.nodes`: lon, lat. The row number is the node id.
  - `.arcs`: from_node, to_node. The row number is the arc id. Arcs are **directed**: a two-way street is two arcs, u→v and v→u.
  - `.route`: the ground-truth **ordered list of arc ids**. It is *not* one label per GPS point.
- **Each trip has its own local road network.** The ids are per trip, not global.
- **Trips are spread worldwide:** 25 UTM zones, both hemispheres (Europe, Russia, the USA, Ecuador…). See the `eda_all_tracks.png` figure.
- **Data quality** (`experiments/dataset_inventory.csv`, `candidate_stats.csv`):
  - No NaNs, timestamps strictly increasing, every route id valid.
  - 69 gaps longer than 60 s across train. The worst are trip 90 (153,064 s, about 42 h) and trip 78 (52,810 s).
  - Train trip 54: most GPS points are more than 50 m from any road (GPS off the network). Train trip 33: the reference route is offset about 120 m from the GPS. Both are label anomalies.
  - Network: no isolated nodes. The largest weakly connected component holds about 97% of nodes. About 87% of arcs have a reverse twin.

## 2. Preprocessing
1. **Projection.** lon/lat are angles, so Euclidean distance in degrees is wrong: 1° of longitude is about 111 km at the equator and about 75 km at 47°N. We project **each trip into its own UTM zone**, picked from the trip's mean lon/lat. After that x/y are metres, and the distortion is under 0.1% inside a zone. We chose per-trip zones because the trips span the globe; one fixed zone breaks, e.g. trip 57 is in Ecuador.
2. **Stationary filter (3 m).** When the car is parked, GPS jitters around one spot and adds noise but no route information. We skip a point if it is less than 3 m from the last kept point. This keeps about 95–97% of points.
3. **Gap split (60 s).** If two consecutive points are more than 60 s apart, we can't know the route between them, so the HMM restarts there.

## 3. Candidate generation and labels (option A)
- **Candidates:** for each GPS point, every arc within **R = 50 m**, keeping the **K = 8** nearest. We find them with a shapely STRtree spatial index, which is fast even for 300k arcs.
  - **Why 50 m?** The EDA histogram of GPS-to-route distance has a median of 5.4 m and a 90th percentile of 19 m. 50 m covers almost all real points.
  - **Result:** the candidate miss rate is **0.4%** on student_test. Train is about 11%, but almost all of that is the two broken trips, 54 and 33.
- **Label (option A, "nearest route arc"):** a point's true arc is the route arc physically closest to it. The candidate row with that arc gets label 1; the others get 0.
  - **Tie-break:** if the car drove a street both ways, both twin arcs are on the route with identical geometry. In that case we pick the one pointing in the direction of travel. Without this, about 8% of labels pointed backwards; with it, 4–5%.
  - **Limitations:**
    - At junctions the "nearest" arc can be the next or previous arc.
    - Heading is noisy at low speed.
    - The reference route itself can be wrong (trip 33).
- **Training rows:** only points whose true arc is among their candidates. A point without one has no correct row, so it would only add misleading negatives. For speed we keep every 3rd point, because consecutive 1 Hz points are near-duplicates.
- **Evaluation:** uses ALL points. A miss or a point with no candidates counts as wrong.

## 4. Features (one row per point–candidate pair)
| Feature | Why |
|---|---|
| `dist_m` | GPS noise is mostly small, so a closer arc is more likely |
| `dist_rank` | relative order matters more than the absolute distance |
| `dist_minus_min` | how much worse this arc is than the best one, which handles noisy areas |
| `heading_diff_deg` | **the only feature that separates a street's two directions**, and also separates side streets at junctions |
| `arc_len_m` | short arcs (junction pieces) are easy to confuse |
| `speed_mps` | heading is unreliable at low speed; the model can learn to trust it less |
| `n_candidates` | dense areas (junctions) are harder |
| `frac_along` | a projection that falls at the very end of an arc suggests the next or previous arc |

## 5. Models (see `experiments/model_comparison.csv`, `reports/figures/model_comparison.png`)
- **Task:** binary classification, "is this candidate the right arc?". At inference, the candidate with the highest probability per point gives the Top-1 prediction.
- **Protocol:**
  - 5-fold **GroupKFold with trip as the group**, so the same trip is never in both training and validation. That avoids leakage between neighbouring, near-identical points.
  - Then refit on all train rows and test on the 10 student_test trips.
- **Class imbalance:** about 17% positives, so we used `class_weight="balanced"` where supported.
- **Models:**
  - LogReg: linear baseline
  - GaussianNB: probabilistic
  - KNN: instance-based, k=25, distance-weighted, scaled
  - Random Forest: bagging
  - LightGBM: boosting
  - MLP: neural network, 64-32 with early stopping
- **Results** (`model_comparison.csv`):

| Model | Family | CV macro F1 | Student-test Top-1 | Student-test macro F1 | Log loss | Continuity | Train (s) |
|---|---|---|---|---|---|---|---|
| LogReg | Linear | 0.837 ± 0.010 | 0.840 | 0.844 | 0.244 | 0.911 | 0.2 |
| GaussianNB | Probabilistic | 0.828 ± 0.010 | 0.832 | 0.823 | 0.487 | 0.896 | 0.1 |
| KNN | Instance-based | 0.887 ± 0.013 | 0.843 | 0.904 | 0.182 | 0.914 | 0.2 (but slowest inference) |
| Random Forest | Bagging | 0.880 ± 0.015 | 0.850 | 0.884 | 0.172 | 0.916 | 18.0 |
| LightGBM | Boosting | 0.877 ± 0.016 | 0.856 | 0.879 | 0.177 | 0.920 | 2.4 |
| **MLP** | Neural net | **0.892 ± 0.013** | 0.849 | **0.905** | **0.129** | 0.917 | 14.0 |

- **Selection:** MLP. It had the best CV macro F1 (the primary metric) and the best log loss, i.e. the best-calibrated probabilities. That matters because the HMM uses the probabilities directly as emission scores. Top-1 accuracy was within 0.5 points across RF, LightGBM and MLP.
- **Over/underfitting:**
  - Compare CV F1 with student_test F1; they are close for all models, so there's no big overfit.
  - The MLP's loss curve plateaus, and early stopping halts training when validation stops improving.
  - LogReg and NB underfit, because their decision boundaries are too simple for feature interactions such as heading × speed.

## 6. Bayesian hyperparameter optimisation (see `experiments/optuna/trials.csv`, `tuned_vs_untuned.csv`)
- **Optuna TPE sampler.** It models which regions of the search space gave good scores and samples the next trial where improvement is most likely. This beats grid or random search when trials are expensive.
- **Search space (MLP):**
  - hidden layers ∈ {32, 64, 64-32, 128-64, 128-64-32}
  - alpha (L2) ∈ [1e-5, 1e-1], log scale
  - learning_rate_init ∈ [1e-4, 1e-2], log scale
  - batch_size ∈ {128, 256, 512, 1024}
- **Objective:** mean 5-fold GroupKFold macro F1 on the train trips only.
- **Trials:** 25, seed 42. The search took 1,839 s, about 31 min.
- **Best:** trial 23, CV macro F1 **0.8926**, with:
  - hidden layers 128-64-32
  - alpha 3.7e-4
  - learning rate 0.0096
  - batch size 128
- **Tuned vs untuned** (same folds, same student_test evaluation):

| | CV macro F1 | Student-test Top-1 | Student-test macro F1 | Log loss | Inference |
|---|---|---|---|---|---|
| Untuned (64-32) | 0.8921 ± 0.0128 | 0.8485 | 0.9048 | 0.1291 | 0.26 ms / 1k rows |
| Tuned (128-64-32) | 0.8926 ± 0.0115 | 0.8492 | 0.9032 | 0.1285 | 0.72 ms / 1k rows |

- **Honest reading:**
  - Tuning gave a very small gain: CV F1 +0.0005, slightly lower variance across folds, slightly better log loss and Top-1.
  - Student-test macro F1 dropped by 0.0016, which is within noise.
  - Conclusion: the default MLP was already near the ceiling for these features. The remaining errors come from label ambiguity (junctions, two-way streets), not hyperparameters.
  - We kept the tuned model because it wins on the search objective and has better-calibrated probabilities (log loss), which helps the HMM.
- **Limitation of the search:** epochs were capped at 60 to keep trials affordable. Trials that drew a very small learning rate hit the cap before converging (ConvergenceWarning in the log), so they are slightly undertrained, not necessarily bad settings.

## 7. HMM and Viterbi (`src/hmm/viterbi.py`, `experiments/hmm_comparison.csv`)
- **Why:** classifying each point on its own can jump between unconnected roads. The HMM adds sequence information and road connectivity.
- **Hidden states:** the candidate arcs at each GPS point.
- **Emission:** log P(arc | GPS point) = the log of the classifier's probability.
- **Transition:** log T = −|d_route − d_gps| / β.
  - d_gps is the straight-line distance between consecutive GPS points.
  - d_route is the driving distance along the network: on the same arc, the distance along it; otherwise the rest of arc a + Dijkstra shortest path from v_a to u_b + the start of arc b.
  - A realistic move has d_route ≈ d_gps, so the penalty is about 0. A detour or impossible jump gets a large penalty. An unreachable move (no path within 2 km, or driving backwards on the same arc) is impossible.
- **Viterbi recurrence:** score_i(b) = log e_i(b) + max_a [score_{i−1}(a) + log T(a→b)]. Store the argmax pointers, then trace back from the best final state.
- **Breaks:**
  - The chain restarts at gaps longer than 60 s.
  - It also restarts at points with no candidates; those points get no prediction.
  - When no candidate is reachable from any previous one ("disconnected road segments"), we restart the chain instead of forcing a jump.
- **Speed:** Dijkstra runs only on roads within 300 m of the trip, with results cached per source node.
- **β:** chosen on student_test (allowed by the guidelines). See `hmm_beta_search.csv`. A small β makes the route stricter (higher continuity); a large β trusts the classifier more.
- **β search on student_test:**

| β (m) | 2 | 5 | 10 | 20 | **50** | 100 |
|---|---|---|---|---|---|---|
| Top-1 | 0.828 | 0.846 | 0.850 | 0.854 | **0.856** | 0.855 |
| Continuity | 0.974 | 0.974 | 0.973 | 0.970 | 0.963 | 0.955 |

  We picked β = 50 (best Top-1). This is a trade-off: a smaller β gives more continuity but lower accuracy.
- **Result (tuned MLP, student_test, all 18,426 cleaned points):**

| | Top-1 accuracy | Route continuity |
|---|---|---|
| Classifier only | 0.8492 | 0.9176 |
| Classifier + HMM | **0.8561** | **0.9633** |

  Continuity improved on all 10 trips, and Top-1 improved on 9 of 10.
- **Where the HMM didn't help:**
  - Trip 56: Top-1 fell from 0.842 to 0.833, even though continuity rose from 0.886 to 0.954. The HMM forces a drivable path, and when the classifier is confidently wrong over a stretch, the path follows the wrong (but connected) road.
  - Trip 77: barely changed (0.757 → 0.760). Most of its errors are the reverse-direction twin, plus slow, noisy sections where heading is unreliable.
- **Error breakdown before the label fix** (useful story): 59% of errors were the reverse twin of the true arc. That led us to the direction-aware tie-break in the labels.

## 8. API and UI
- **Laptop 1:** `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
  - `GET /trips` lists folders in `data/evaluator_test/`, falling back to `student_test` for rehearsal.
  - `POST /predict {trip_id}` runs the pipeline:
    1. read .track/.nodes/.arcs
    2. clean
    3. generate candidates and features
    4. MLP probabilities
    5. Viterbi
    6. `api/evaluate.py` scores the result against `.route`
  - **Only `api/evaluate.py` reads `.route`.** Ground truth never goes to the UI; only the numbers do.
  - Each request is logged in the terminal.
- **Laptop 2:** `streamlit run ui/app.py`. It needs only the `ui/` folder and 4 packages. Enter the API address, pick a trip, run it, and it shows a Folium map (GPS in blue, prediction in red) and the metrics.
- **Metrics shown in the UI:**
  - Point Top-1 accuracy
  - Route continuity (the share of consecutive predictions that are the same arc or directly connected)
  - Route-arc precision and recall: the predicted set of arcs compared with the reference route (sequence-level)

## 9. Likely tricky questions
- *Why not Euclidean distance on lon/lat?* See §2.1.
- *Why GroupKFold?* Consecutive points from one trip are nearly identical. A random split would leak them, and CV would be over-optimistic.
- *Why is training accuracy not used for selection?* It rewards memorising and says nothing about unseen trips.
- *Why macro F1?* The classes are imbalanced (17% positives). Accuracy would look good just by predicting "wrong arc" everywhere. Macro F1 weighs both classes equally.
- *What does log loss tell you?* Probability quality. It matters because the HMM multiplies these probabilities.
- *Could evaluator data leak?* No. It isn't in training. Decoder β was chosen on student_test. The prediction code (`src/pipeline.py`) never reads `.route`.
- *Biggest limitation?* The option-A labels are approximate at junctions and on out-and-back roads. Heading is noisy at low speed. The reference route is sometimes wrong (trip 33).

## 10. API results on all 10 student_test trips (tuned MLP + HMM, β = 50)
| Trip | Top-1 | Continuity | Route-arc precision | Route-arc recall | Time |
|---|---|---|---|---|---|
| 00000000 | 0.946 | 0.990 | 0.851 | 0.922 | 0.34 s |
| 00000007 | 0.906 | 0.998 | 0.960 | 0.984 | 0.16 s |
| 00000009 | 0.835 | 0.929 | 0.894 | 0.825 | 0.25 s |
| 00000040 | 0.929 | 0.984 | 0.895 | 0.968 | 0.39 s |
| 00000052 | 0.973 | 0.962 | 1.000 | 0.863 | 0.05 s |
| 00000056 | 0.833 | 0.954 | 0.954 | 0.886 | 0.78 s |
| 00000065 | 0.863 | 0.964 | 0.894 | 0.876 | 0.17 s |
| 00000076 | 0.814 | 0.935 | 0.783 | 0.832 | 0.52 s |
| 00000077 | 0.760 | 0.968 | 0.720 | 0.951 | 0.62 s |
| 00000091 | 0.832 | 0.948 | 0.905 | 0.784 | 0.17 s |

These match `experiments/hmm_comparison.csv` exactly, which confirms that the API runs the same pipeline as the offline evaluation.
Final maps (GPS / network / reference / predicted): `maps/final_00000000.html`, `final_00000076.html`, `final_00000077.html`.
