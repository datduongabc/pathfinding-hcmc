import os
import random

import networkx as nx
import osmnx as ox
import pytest

from src import heuristic, traffic_model
from src.astar import NoPathFoundError, find_path

GRAPH_PATH = "data/graph.graphml"

# Three collinear points (same longitude, varying latitude) used throughout
# this file. Collinearity makes the true cost-to-go a closed-form sum of two
# straight legs, so admissibility and consistency can be checked exactly
# instead of by searching.
POINT_A = (10.80, 106.70)
POINT_B = (10.78, 106.70)
POINT_GOAL = (10.70, 106.70)
DIST_AB_KM = heuristic.haversine_km(*POINT_A, *POINT_B)
DIST_B_GOAL_KM = heuristic.haversine_km(*POINT_B, *POINT_GOAL)
DIST_A_GOAL_KM = heuristic.haversine_km(*POINT_A, *POINT_GOAL)
V_FREE_BOUND = 80.0  # motorway class, matches the synthetic edges below


def naive_heuristic_hours(lat, lon, goal_lat, goal_lon, hour, v_free_bound_kmh):
    """A naive heuristic that bounds speed by the multiplier at `hour` only,
    not the best one within the look-ahead window. Kept as a foil: it
    overestimates when traffic clears up mid-trip, which shows why the
    windowed version is needed."""
    dist_km = heuristic.haversine_km(lat, lon, goal_lat, goal_lon)
    m_t = traffic_model.time_of_day_multiplier(hour)
    return dist_km / (v_free_bound_kmh * m_t)


def _true_cost_to_go(departure_hour):
    """True A->B->Goal travel time on the synthetic collinear points,
    using each edge's actual traversal-time multiplier (not the departure
    time's)."""
    cost_ab = traffic_model.edge_travel_time_hours(
        DIST_AB_KM * 1000.0, "motorway", None, departure_hour, {})
    arrival_hour = departure_hour + cost_ab
    cost_b_goal = traffic_model.edge_travel_time_hours(
        DIST_B_GOAL_KM * 1000.0, "motorway", None, arrival_hour, {})
    return cost_ab + cost_b_goal


def _consistency_gap(point_from, point_to, goal, departure_hour, highway="motorway"):
    """h(from,t) - (cost(from,to,t) + h(to,t')). Positive means the
    consistency inequality is violated at this departure hour."""
    dist_km = heuristic.haversine_km(*point_from, *point_to)
    cost = traffic_model.edge_travel_time_hours(dist_km * 1000.0, highway, None, departure_hour, {})
    arrival_hour = departure_hour + cost
    h_from = heuristic.heuristic_hours(*point_from, *goal, departure_hour, V_FREE_BOUND)
    h_to = heuristic.heuristic_hours(*point_to, *goal, arrival_hour, V_FREE_BOUND)
    return h_from - (cost + h_to)


def test_collinear_points_are_set_up_correctly():
    assert DIST_AB_KM == pytest.approx(2.224, abs=0.01)
    assert DIST_B_GOAL_KM == pytest.approx(8.895, abs=0.01)
    assert DIST_A_GOAL_KM == pytest.approx(DIST_AB_KM + DIST_B_GOAL_KM, abs=0.001)


def test_windowed_heuristic_is_admissible_everywhere_naive_one_is_not():
    """Scanning a full day: the windowed heuristic never overestimates the
    true cost-to-go, but the naive (unwindowed) one does somewhere -
    which is the failure the window exists to prevent."""
    hours = [h * 0.5 for h in range(48)]  # every 30 minutes across a day
    naive_violated_somewhere = False
    for departure_hour in hours:
        true_cost = _true_cost_to_go(departure_hour)
        windowed_h = heuristic.heuristic_hours(*POINT_A, *POINT_GOAL, departure_hour, V_FREE_BOUND)
        naive_h = naive_heuristic_hours(*POINT_A, *POINT_GOAL, departure_hour, V_FREE_BOUND)

        assert windowed_h <= true_cost + 1e-9, f"windowed heuristic inadmissible at hour={departure_hour}"
        if naive_h > true_cost + 1e-9:
            naive_violated_somewhere = True

    assert naive_violated_somewhere, "expected the naive unwindowed heuristic to be inadmissible somewhere in the day"


def test_consistency_holds_in_the_interior_of_a_plateau():
    # 11h-14h departures sit deep inside the flat 10-16h plateau, far from
    # any window-edge effect; consistency should hold with margin.
    for departure_hour in [11.0, 12.0, 13.0, 14.0]:
        gap = _consistency_gap(POINT_A, POINT_B, POINT_GOAL, departure_hour)
        assert gap <= 1e-9, f"unexpected consistency violation at hour={departure_hour}"


def test_consistency_can_fail_somewhere_in_the_day():
    # The documented exception: consistency is not guaranteed everywhere.
    # At least one hour in a full day must violate it (the window's forward
    # edge sliding onto a rising anchor), even though most hours hold.
    # For these synthetic points the violation window is narrow (only
    # about 4 minutes wide, around t=16h37m30s where M_W's left edge and
    # 3-hours-ahead right edge cross), so a half-hour grid can straddle it
    # entirely and miss it; sample at 1-minute resolution instead.
    hours = [h / 60.0 for h in range(24 * 60)]
    gaps = [_consistency_gap(POINT_A, POINT_B, POINT_GOAL, h) for h in hours]
    assert any(gap > 1e-9 for gap in gaps), "expected at least one consistency violation somewhere in the day"
    assert any(gap <= 1e-9 for gap in gaps), "expected consistency to hold somewhere in the day too"


def test_compute_free_flow_bound_kmh_picks_fastest_class_present():
    g = nx.MultiDiGraph()
    g.add_node("A", y=10.8, x=106.7)
    g.add_node("B", y=10.79, x=106.7)
    g.add_edge("A", "B", highway="secondary")
    assert heuristic.compute_free_flow_bound_kmh(g) == traffic_model.free_flow_speed_kmh("secondary")


def test_compute_free_flow_bound_kmh_defaults_on_empty_graph():
    g = nx.MultiDiGraph()
    assert heuristic.compute_free_flow_bound_kmh(g) == traffic_model.DEFAULT_FREE_FLOW_SPEED_KMH


@pytest.mark.skipif(not os.path.exists(GRAPH_PATH), reason="cached graph not present")
def test_randomized_admissibility_spot_check_on_real_graph():
    """Random start/goal pairs on the real road graph: the heuristic at the
    start must not exceed the travel time A* actually finds."""
    graph = ox.load_graphml(GRAPH_PATH)
    v_free_bound = heuristic.compute_free_flow_bound_kmh(graph)
    nodes = list(graph.nodes(data=True))
    rng = random.Random(42)

    checked = 0
    for _ in range(20):
        (start_id, start_data), (goal_id, goal_data) = rng.sample(nodes, 2)
        hour = rng.uniform(0.0, 24.0)
        try:
            # Empty bias keeps this independent of the committed TomTom data.
            result = find_path(graph, start_id, goal_id, hour,
                               v_free_bound_kmh=v_free_bound, corridor_bias={})
        except NoPathFoundError:
            # Random node pairs can land in different connected components.
            # Any other exception is a real failure and must surface.
            continue
        h = heuristic.heuristic_hours(
            start_data["y"], start_data["x"], goal_data["y"], goal_data["x"], hour, v_free_bound)
        assert h <= result.total_travel_time_hours + 1e-6
        assert result.total_travel_time_hours < traffic_model.WINDOW_HOURS, (
            "sampled path exceeded the heuristic's W-hour assumption")
        checked += 1

    assert checked >= 10
