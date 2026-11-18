# PathSnap UI — Laptop 2 setup

Only this `ui/` folder is needed on Laptop 2.

```bash
git clone <repo-url> && cd <repo>/ui
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-ui.txt
streamlit run app.py
```

In the sidebar, enter `http://<Laptop-1-IP>:8000`. To find the IP on Laptop 1, run `ipconfig getifaddr en0` on macOS or `ipconfig` on Windows.
Both laptops must be on the same network. If college Wi-Fi blocks laptop-to-laptop traffic, use a phone hotspot.
