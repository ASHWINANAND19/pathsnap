"""PathSnap evaluator UI (runs on Laptop 2).

Self-contained: needs only this folder + requirements-ui.txt. It never sees ground truth —
it sends a trip id to the API and draws what comes back.

Run:  streamlit run app.py
"""
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap — map-matching evaluator")

api = st.sidebar.text_input("API address (Laptop 1)", os.environ.get("PATHSNAP_API", "http://localhost:8000")).rstrip("/")
st.sidebar.caption("On Laptop 1 run `uvicorn api.main:app --host 0.0.0.0 --port 8000`, "
                   "then enter http://<laptop-1-IP>:8000 here.")


@st.cache_data(ttl=30, show_spinner=False)
def get_trips(url):
    r = requests.get(f"{url}/trips", timeout=5)
    r.raise_for_status()
    return r.json()


try:
    info = get_trips(api)
except requests.RequestException as e:
    st.error(f"Cannot reach the API at {api}. Check the address, that the API is running, and that both "
             f"laptops are on the same network (try a phone hotspot if college Wi-Fi blocks it).\n\n{e}")
    st.stop()

st.sidebar.success(f"Connected — {len(info['trips'])} trips ({info['source']})")
trip_id = st.selectbox("Choose an evaluation trip", info["trips"])

if st.button("Run map-matching", type="primary"):
    with st.spinner("API is predicting..."):
        try:
            r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=300)
            r.raise_for_status()
            st.session_state["result"] = r.json()
        except requests.RequestException as e:
            st.error(f"Prediction failed: {e}")

res = st.session_state.get("result")
if res and res["trip_id"] == trip_id:
    m = res["metrics"]
    fmt = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
    c = st.columns(5)
    c[0].metric("Point accuracy (Top-1)", fmt(m.get("top1_accuracy")))
    c[1].metric("Route continuity", fmt(m.get("route_continuity")))
    c[2].metric("Route arc precision", fmt(m.get("route_arc_precision")))
    c[3].metric("Route arc recall", fmt(m.get("route_arc_recall")))
    c[4].metric("API time", f"{res['time_ms'] / 1000:.1f} s")
    st.caption(f"{res['n_gps_points']} GPS points, {res['n_matched_points']} matched after cleaning. "
               "Metrics are computed on the API side against its internal reference data.")

    gps = res["gps"]
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=14, tiles="OpenStreetMap")
    g1 = folium.FeatureGroup(name="Raw GPS points (blue)")
    for p in gps:
        folium.CircleMarker(p, radius=2, color="blue", fill=True, weight=0, fill_opacity=0.7).add_to(g1)
    g2 = folium.FeatureGroup(name="Predicted route (red)")
    for seg in res["pred_route"]:
        folium.PolyLine(seg, color="red", weight=4, opacity=0.85).add_to(g2)
    g1.add_to(fmap); g2.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)], [max(p[0] for p in gps), max(p[1] for p in gps)]])
    st_folium(fmap, height=600, use_container_width=True, returned_objects=[])
