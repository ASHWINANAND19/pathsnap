# PathSnap: problem brief

## Timeline (today, 18 Nov 2026)
| Time | What |
|---|---|
| 8:00 | Problem released, start working |
| **12:00–1:00** | **First evaluation**: initial deliverable (EDA + early models) |
| 2:45–3:15 | CBT quiz for AIML students (covers the hackathon topic + all ML labs). Attendance is mandatory. |
| **before 4:00** | **Submit**: public GitHub repo + one PDF report |
| 4:00–5:00 | Final demo + viva |

## The task
GPS traces are noisy. A car on road A can show up on a parallel road B. Given each GPS point,
work out **which road segment (arc) the car was actually on**, and reconstruct a connected route.

## Data (verified)
- 80 train trips, 10 student-test trips. A further 10 evaluator trips are held back for the demo.
- Each trip has its own local road graph:
  - `.track`: lon, lat, t (~1 Hz). Median about 1.5k points per trip, max 19k.
  - `.nodes`: lon, lat. The row index is the node id.
  - `.arcs`: from_node, to_node. The row index is the arc id.
  - `.route`: the ground-truth **sequence of arc ids**. It is *not* one label per GPS point.
- Region: about 11°E, 47.6°N (southern Bavaria/Tyrol). Use a UTM projection (EPSG:32632) or haversine, not raw degrees.
- No NaNs, timestamps are monotonic, and every route arc id is valid. Some trips have big gaps: one train trip has a 153,064 s gap. See `experiments/dataset_inventory.csv`.

## The key modelling step you must design
The route gives an ordered list of arcs, not a label for each point. So you must:
1. **Generate candidates:** for each GPS point, take the arcs within R metres (or the k nearest).
2. **Label the candidates:** mark the candidate that is the "true" arc as 1 and the rest as 0. The true arc is the route arc closest to the point, kept consistent with the route's order.
3. **Engineer features for each (point, candidate) pair:** perpendicular distance, heading difference, distance rank, arc length, speed, and so on.
4. **Train a binary classifier** that scores the candidates. Compare at least 5 models from different families (for example LogReg, KNN, RF, HistGB/LightGBM, MLP) and use **GroupKFold by trip**.
5. **Tune the best model with Bayesian optimisation** (Optuna TPE) and keep the record of every trial.
6. **Run HMM + Viterbi:**
   - The hidden states are the candidate arcs at each point.
   - The emission probability is the classifier's probability.
   - The transition probability comes from graph connectivity and network distance compared with GPS distance.
   - Then compare classifier-only results with classifier+HMM results.

## Metrics to report
- Observation-level Top-1 accuracy, macro F1, log loss
- Candidate miss rate
- Route continuity
- Training and inference time

## System for the demo
- **Laptop 1:** a FastAPI server holding the model, the HMM, the road graphs and the **ground truth (server-side only)**. Show its request logs in the terminal.
- **Laptop 2:** the UI. It lists the 10 evaluator trips and sends the chosen trip id. It then shows the predicted route on a Folium map and the accuracy the API returns.
- It must work for all 10 trips. No hardcoded predictions.

## Integrity rules (they will check)
- Your weights, configs and code are compared across teams.
- Reports are checked for AI-generated text.
- Every number and graph must come from a real run.
- Every team member must be able to explain everything in the viva.
