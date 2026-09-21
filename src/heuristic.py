"""Admissible, time-variant straight-line heuristic for time-dependent A*."""
import math

from . import traffic_model


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_free_flow_bound_kmh(graph):
    """Fastest free-flow speed among all edges actually present in `graph`
    (an OSMnx/networkx MultiDiGraph; each edge has a `highway` tag)."""
    best = 0.0
    for _, _, data in graph.edges(data=True):
        v = traffic_model.free_flow_speed_kmh(data.get("highway"))
        if v > best:
            best = v
    if best <= 0.0:
        return traffic_model.DEFAULT_FREE_FLOW_SPEED_KMH
    return best


def heuristic_hours(lat, lon, goal_lat, goal_lon, hour, v_free_bound_kmh,
                     window_hours=traffic_model.WINDOW_HOURS):
    """h(n, t): admissible lower bound on remaining travel time, in hours.

    Bounds speed by the worst (highest) time-of-day multiplier reachable
    within `window_hours` of `hour`, not just at `hour` itself - see the
    Heuristic section of
    docs/superpowers/specs/2026-09-16-hcmc-pathfinding-design.md for why
    using m_time(t) alone is not admissible.
    """
    dist_km = haversine_km(lat, lon, goal_lat, goal_lon)
    m_bound = traffic_model.max_time_multiplier_over_window(hour, window_hours)
    denom = v_free_bound_kmh * m_bound
    if denom <= 0:
        return float("inf")
    return dist_km / denom
