# HCMC Time-Variant A* Pathfinding

Given a real map of the city and two points of interest, the system finds the fastest route between them.
It uses **A\* search** over a road-network graph whose edge costs change with the time of day. The result is rendered on an interactive map. The search heuristic's **admissibility** and **consistency** are explained in the notebook and checked empirically by the test suite.

## What this covers

| Assignment requirement | Status |
|---|---|
| Map of at least 100 points of interest | 150 real, named POIs (hospitals, schools, markets, temples, hotels, museums, parks, and more) |
| Time-variant heuristic (function of density/speed/flow, not a constant) | Straight-line heuristic bounded by a traffic model that varies by time of day |
| Shortest/cheapest path via A\* search | Custom time-dependent A\* |
| Visualization on a map | Interactive `folium` map: road network, POIs, and the computed route |
| Admissibility / consistency analysis | Verified by the test suite against real data |
| Real-time data | Traffic model calibrated against cached TomTom Traffic Flow API snapshots |

## Project layout

```text
src/
  astar.py                 # Custom time-dependent A* search
  calibration.py           # TomTom Traffic Flow API calibration
  data_collection.py       # OSM road network + 150-POI download, caching, category tagging
  heuristic.py             # Admissible, time-variant A* heuristic
  traffic_model.py         # Time-of-day / corridor congestion model, edge cost
  visualization.py         # folium map rendering
tests/                     # pytest suite, one file per src module
data/                      # Cached road graph, POIs, and calibration data (committed)
Pathfinding_HCMC.ipynb     # Narrative notebook: setup, explanation, interactive demo
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Docker

A container is provided if you'd rather not set up a local Python environment.

```bash
docker build -t hcmc-pathfinding .
docker run -p 8888:8888 hcmc-pathfinding
```

Then open the URL printed in the container logs.

## Running the notebook

The road graph and 150 POIs are cached under `data/` and committed to the repo.
The notebook therefore runs fully offline.

```bash
jupyter notebook Pathfinding_HCMC.ipynb
```

Open the notebook and run all cells.
Then use the "Find Path" widget to pick any two of the 150 POIs and a departure time.
The route is computed and drawn live.

## Refreshing traffic data

The traffic model reads its real-world speed data from `data/tomtom_calibration.json`.

### What it does

`src/calibration.py` asks the TomTom Traffic Flow API for the current speed on 12 major corridors in the map area (Điện Biên Phủ, Nguyễn Thị Minh Khai, Trần Hưng Đạo, and others).
For each corridor it stores the ratio `current speed / free-flow speed`.
The snapshot is stamped with the current HCMC local hour (UTC+7) and **appended** to the JSON file.
Existing snapshots are never overwritten.

At load time, `traffic_model.load_corridor_bias()` divides each ratio by the model's time-of-day multiplier for that hour.
The averaged result becomes a per-corridor bias, clamped to [0.5, 1.0].
More snapshots taken at different hours give a better fit.

### Steps

1. Get a free API key at https://developer.tomtom.com/
2. Copy `.env.example` to `.env`.
3. Set `TOMTOM_API_KEY=<your key>` in `.env`.
4. Run the script from the project root with the virtual environment active:

   ```bash
   python -m src.calibration
   ```

5. Repeat at different times of day.
   Good slots are midday (around 13:00), evening rush (around 17:30-18:00), after rush (around 20:30), and late night (around 23:00).
6. Check the result: `data/tomtom_calibration.json` should have one more entry under `snapshots`, each with a `captured_at`, an `hour`, and 12 corridors.

### After a refresh

- The notebook loads the calibration once, when its setup cell runs.
  Restart the kernel and run all cells, or the old numbers stay in memory.
- Run `pytest -v` to confirm the model still behaves.
- Commit the updated JSON file if you want the new data to ship with the project.
