"""One-time TomTom Traffic Flow API calibration: fetch real speed ratios for
known HCMC corridors and cache them for traffic_model.py to fit against."""
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Read TOMTOM_API_KEY from a local .env if present (see README) - without
# this, a key placed in .env would be silently ignored.
load_dotenv()

# Ho Chi Minh City is UTC+7 year-round (no DST). traffic_model.py's
# time-of-day model is expressed in HCMC local hours, so snapshots must be
# stamped with the local hour or every corridor bias is fit against the
# wrong m_time(hour).
HCMC_TZ = timezone(timedelta(hours=7))

FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

# Fixed sample points on major corridors inside the bounding box, one
# TomTom request each.
#
# Keys must match the edge `name` in data/graph.graphml exactly, including
# diacritics and any prefix OSM stores (e.g. "Đường 3 Tháng 2"), because
# traffic_model.corridor_multiplier looks streets up by exact string. A key
# that doesn't match is not an error, the calibration just never applies.
CORRIDORS = {
    "Điện Biên Phủ": (10.7893, 106.6923),
    "Cách Mạng Tháng Tám": (10.7769, 106.6883),
    "Nguyễn Văn Cừ": (10.7592, 106.6822),
    "Võ Văn Kiệt": (10.7472, 106.6934),
    "Hai Bà Trưng": (10.7852, 106.7008),
    "Nam Kỳ Khởi Nghĩa": (10.7797, 106.6947),
    "Lý Thường Kiệt": (10.7724, 106.6588),
    "Xô Viết Nghệ Tĩnh": (10.8015, 106.7099),
    "Đường 3 Tháng 2": (10.7679, 106.6688),
    "Nguyễn Thị Minh Khai": (10.7808, 106.6939),
    "Trần Hưng Đạo": (10.7547713, 106.6789487),
    "Nguyễn Trãi": (10.7530389, 106.6614172),
}


def _fetch_one_corridor(http, api_key, lat, lon, name):
    """Call the TomTom Flow Segment Data endpoint for one corridor point.

    Returns a result dict, or None if the response shape is unexpected
    (logged as a warning, not fatal). Raises requests.RequestException (via
    raise_for_status) on HTTP failure.
    """
    resp = http.get(FLOW_URL, params={"point": f"{lat},{lon}", "key": api_key}, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("flowSegmentData", {})
    current = data.get("currentSpeed")
    free_flow = data.get("freeFlowSpeed")
    if current is None or free_flow is None or free_flow == 0:
        logger.warning("Unexpected TomTom response for %s: %s", name, data)
        return None
    return {
        "lat": lat,
        "lon": lon,
        "current_speed_kmh": current,
        "free_flow_speed_kmh": free_flow,
        "confidence": data.get("confidence"),
        "ratio": current / free_flow,
    }


def fetch_corridor_snapshot(api_key, corridors=None, session=None):
    """Call the TomTom Flow Segment Data endpoint for each corridor point.

    Returns {"captured_at": iso8601, "hour": float, "corridors": {name: {...}}}.
    `hour` is the HCMC *local* hour-of-day (UTC+7), matching the convention
    traffic_model.time_of_day_multiplier expects; `captured_at` keeps the
    explicit +07:00 offset so the snapshot stays self-documenting.

    Requests run concurrently, one thread per corridor: they are independent,
    and sequentially a full snapshot could take minutes if the API is slow.
    A `session` you pass in is shared across those threads, but
    `requests.Session` is not documented as thread-safe, so pass one only if
    you know that is safe. The default (module-level `requests`) opens a
    separate connection per call.

    Raises requests.RequestException on HTTP failure. A response with an
    unexpected shape skips that corridor instead of failing the snapshot.
    """
    corridors = corridors or CORRIDORS
    http = session or requests
    now = datetime.now(HCMC_TZ)

    result = {}
    with ThreadPoolExecutor(max_workers=len(corridors) or 1) as pool:
        future_to_name = {
            pool.submit(_fetch_one_corridor, http, api_key, lat, lon, name): name
            for name, (lat, lon) in corridors.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            entry = future.result()  # re-raises the request's exception here, if any
            if entry is not None:
                result[name] = entry

    return {
        "captured_at": now.isoformat(),
        "hour": now.hour + now.minute / 60.0,
        "corridors": result,
    }


def calibrate(output_path="data/tomtom_calibration.json", api_key=None, session=None):
    """Fetch one snapshot and append it to the cached calibration file.

    Meant to be run manually a few times a day during development (see
    README) - never called automatically at notebook runtime.
    """
    api_key = api_key or os.environ.get("TOMTOM_API_KEY")
    if not api_key:
        raise RuntimeError("TOMTOM_API_KEY is not set; export it or pass api_key=")

    snapshot = fetch_corridor_snapshot(api_key, session=session)

    path = Path(output_path)
    existing = {"snapshots": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict) or not isinstance(loaded.get("snapshots", []), list):
                raise ValueError("calibration file must be a JSON object with a 'snapshots' list")
            existing = loaded
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Existing calibration file at %s is malformed (%s); overwriting", path, exc)
    existing.setdefault("snapshots", []).append(snapshot)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return snapshot


if __name__ == "__main__":
    calibrate()
