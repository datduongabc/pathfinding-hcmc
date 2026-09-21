"""Time-variant traffic speed and edge-cost model for the HCMC road graph."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Floor on effective speed (about walking pace). Without it, a heavily
# congested edge could get a near-zero speed and an unbounded travel time.
MIN_SPEED_KMH = 5.0

# How far ahead the heuristic looks for a faster time of day. The
# admissibility argument only holds for trips shorter than this (astar warns
# otherwise); trips inside central HCMC take well under an hour, so 3h is a
# generous margin. A wider window is safer but weakens the heuristic.
WINDOW_HOURS = 3.0

# Assumed free-flow speeds per OSM road class in dense urban HCMC, not
# measured values. TomTom calibration corrects the residual on major streets.
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

# (hour, multiplier) anchors, linearly interpolated in between. The
# multiplier scales free-flow speed: near 1 means free-flowing, low means
# congested. Free-flowing at night (22-6h), congested in the morning rush
# (7-9h) and evening rush (17-19h), moderate in between.
#
# Speed only recovers between 9-10h and 19-22h. That is why the heuristic
# needs a look-ahead window (see max_time_multiplier_over_window) and why it
# is not strictly consistent in the hours just before those recoveries.
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

# Corridor bias may only slow traffic down, never speed it up. The heuristic
# bounds speed by (fastest free-flow speed x time multiplier), so a bias
# above 1.0 could push a real edge past that bound and make the heuristic
# overestimate. The lower bound keeps one bad snapshot (an accident, a
# closure) from making a street effectively impassable.
CORRIDOR_BIAS_MIN = 0.5
CORRIDOR_BIAS_MAX = 1.0


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

    Used by heuristic.py as the heuristic's speed bound. Using the multiplier
    at `hour` alone would overestimate the remaining time whenever traffic
    clears up later in the trip (e.g. leaving at 8:30 and finishing after
    10:00), so h(n, t) would no longer be admissible.
    """
    n_steps = max(1, int(round(window_hours / step_hours)))
    return max(
        time_of_day_multiplier(hour + i * step_hours)
        for i in range(n_steps + 1)
    )


def load_corridor_bias(calibration_path="data/tomtom_calibration.json"):
    """Fit a per-corridor bias multiplier from cached TomTom snapshots.

    Returns {street_name: bias}: how much slower a street runs than the
    city-wide time-of-day multiplier already predicts, so the two effects
    are not double counted. Falls back to an empty dict (every street gets
    bias 1.0) if the calibration file is missing or malformed, so the
    search still works without calibration data.
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
        bias[street] = max(CORRIDOR_BIAS_MIN, min(CORRIDOR_BIAS_MAX, avg))
    return bias


def corridor_multiplier(street_name, corridor_bias):
    if not street_name:
        return 1.0
    return corridor_bias.get(street_name, 1.0)


def effective_speed_kmh(highway, street_name, hour, corridor_bias):
    # v_free: free-flow speed of the road class. m_time: city-wide slowdown
    # at this hour. m_corr: extra slowdown measured on this specific street.
    v_free = free_flow_speed_kmh(highway)
    m_time = time_of_day_multiplier(hour)
    m_corr = corridor_multiplier(street_name, corridor_bias)
    v_eff = v_free * m_time * m_corr
    return max(v_eff, MIN_SPEED_KMH)


def edge_travel_time_hours(length_m, highway, street_name, hour, corridor_bias):
    v_eff = effective_speed_kmh(highway, street_name, hour, corridor_bias)
    length_km = length_m / 1000.0
    return length_km / v_eff
