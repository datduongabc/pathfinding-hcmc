"""Time-variant traffic speed and edge-cost model for the HCMC road graph."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_SPEED_KMH = 5.0
WINDOW_HOURS = 3.0

FREE_FLOW_SPEED_KMH = {
    "motorway": 80.0,
    "motorway_link": 60.0,
    "trunk": 70.0,
    "trunk_link": 50.0,
    "primary": 50.0,
    "primary_link": 40.0,
    "secondary": 40.0,
    "secondary_link": 30.0,
    "tertiary": 35.0,
    "tertiary_link": 25.0,
    "unclassified": 30.0,
    "residential": 25.0,
    "living_street": 15.0,
    "service": 20.0,
}
DEFAULT_FREE_FLOW_SPEED_KMH = 30.0

# (hour, multiplier) anchors, piecewise-linear between consecutive pairs.
# Low at night (0-6h, 22-24h), high during rush (7-9h, 16-19h; note the
# 16-17h approach to the evening peak is itself decreasing), moderate
# mid-day (10-16h). The only rising stretches of m_time itself are 9-10h
# and 19-22h; the windowed heuristic's consistency-violation windows are
# the ~3h-earlier stretches (6h40m, 7h) and (16h37m30s, 19h), where the
# sliding window's forward edge climbs those rises - see the spec's
# Heuristic section.
TIME_OF_DAY_ANCHORS = [
    (0.0, 0.95),
    (6.0, 0.95),
    (7.0, 0.55),
    (9.0, 0.55),
    (10.0, 0.75),
    (16.0, 0.75),
    (17.0, 0.50),
    (19.0, 0.50),
    (22.0, 0.95),
    (24.0, 0.95),
]

WINDOW_SAMPLE_STEP_HOURS = 1.0 / 12.0  # 5-minute resolution


def free_flow_speed_kmh(highway):
    """Look up free-flow speed for an OSM `highway` tag value (or list of
    values - OSMnx sometimes tags an edge with several; the fastest wins)."""
    if isinstance(highway, (list, tuple, set)):
        speeds = [FREE_FLOW_SPEED_KMH.get(h, DEFAULT_FREE_FLOW_SPEED_KMH) for h in highway]
        return max(speeds) if speeds else DEFAULT_FREE_FLOW_SPEED_KMH
    return FREE_FLOW_SPEED_KMH.get(highway, DEFAULT_FREE_FLOW_SPEED_KMH)


def time_of_day_multiplier(hour):
    """Piecewise-linear time-of-day multiplier; `hour` wraps mod 24."""
    h = hour % 24.0
    anchors = TIME_OF_DAY_ANCHORS
    for (h0, m0), (h1, m1) in zip(anchors, anchors[1:]):
        if h0 <= h <= h1:
            if h1 == h0:
                return m0
            frac = (h - h0) / (h1 - h0)
            return m0 + frac * (m1 - m0)
    return anchors[-1][1]


def max_time_multiplier_over_window(hour, window_hours=WINDOW_HOURS,
                                     step_hours=WINDOW_SAMPLE_STEP_HOURS):
    """Sup of time_of_day_multiplier over [hour, hour + window_hours].

    Used by heuristic.py as the heuristic's speed bound so h(n, t) stays
    admissible even though m_time may rise later in the trip - see the
    "Why a window, and why 3 hours" section of
    docs/superpowers/specs/2026-09-16-hcmc-pathfinding-design.md.
    """
    n_steps = max(1, int(round(window_hours / step_hours)))
    return max(
        time_of_day_multiplier(hour + i * step_hours)
        for i in range(n_steps + 1)
    )


def load_corridor_bias(calibration_path="data/tomtom_calibration.json"):
    """Fit a per-corridor bias multiplier from cached TomTom snapshots.

    Returns {street_name: bias} where bias is in [0.5, 1.0] - the
    residual, street-specific dampening left after factoring out the
    generic time-of-day multiplier already applied by
    time_of_day_multiplier. Falls back to an empty dict (all corridors
    default to bias 1.0) if the calibration file is missing or malformed.
    """
    path = Path(calibration_path)
    if not path.exists():
        logger.warning("No calibration file at %s; corridor bias defaults to 1.0 everywhere", path)
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read calibration file %s (%s); corridor bias defaults to 1.0", path, exc)
        return {}

    residuals = {}
    try:
        for snapshot in raw.get("snapshots", []):
            hour = snapshot.get("hour")
            if hour is None:
                continue
            m_t = time_of_day_multiplier(hour)
            if m_t <= 0:
                continue
            for street, corridor in snapshot.get("corridors", {}).items():
                ratio = corridor.get("ratio")
                if ratio is None:
                    continue
                residuals.setdefault(street, []).append(ratio / m_t)
    except (AttributeError, TypeError) as exc:
        logger.warning("Calibration file %s has unexpected structure (%s); corridor bias defaults to 1.0", path, exc)
        return {}

    bias = {}
    for street, values in residuals.items():
        avg = sum(values) / len(values)
        bias[street] = max(0.5, min(1.0, avg))
    return bias


def corridor_multiplier(street_name, corridor_bias):
    if not street_name:
        return 1.0
    return corridor_bias.get(street_name, 1.0)


def effective_speed_kmh(highway, street_name, hour, corridor_bias):
    v_free = free_flow_speed_kmh(highway)
    m_time = time_of_day_multiplier(hour)
    m_corr = corridor_multiplier(street_name, corridor_bias)
    v_eff = v_free * m_time * m_corr
    return max(v_eff, MIN_SPEED_KMH)


def edge_travel_time_hours(length_m, highway, street_name, hour, corridor_bias):
    v_eff = effective_speed_kmh(highway, street_name, hour, corridor_bias)
    length_km = length_m / 1000.0
    return length_km / v_eff
