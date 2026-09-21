import networkx as nx
import pytest

from src import heuristic, traffic_model
from src.astar import (
    MissingEdgeLengthError, NoPathFoundError, find_path, shortest_parallel_edge)

# Every test passes an explicit (empty) corridor bias. Left to its default,
# find_path reads data/tomtom_calibration.json, which would make these tests
# depend on whichever snapshots happen to be committed.
NO_BIAS = {}


def _add_node(g, node_id, lat, lon):
    g.add_node(node_id, y=lat, x=lon)


def _add_edge(g, u, v, length_m, highway="residential", name=None):
    g.add_edge(u, v, length=length_m, highway=highway, name=name)


def _linear_graph():
    """A -> B -> C, straight line, all residential (25 km/h free-flow)."""
    g = nx.MultiDiGraph()
    _add_node(g, "A", 10.80, 106.70)
    _add_node(g, "B", 10.79, 106.70)
    _add_node(g, "C", 10.78, 106.70)
    _add_edge(g, "A", "B", 1111.0)
    _add_edge(g, "B", "C", 1111.0)
    return g


def test_finds_path_on_simple_linear_graph():
    g = _linear_graph()
    result = find_path(g, "A", "C", departure_hour=2.0, corridor_bias=NO_BIAS)
    assert result.node_path == ["A", "B", "C"]
    assert result.total_travel_time_hours > 0


def test_disconnected_start_end_raises():
    g = nx.MultiDiGraph()
    _add_node(g, "A", 10.80, 106.70)
    _add_node(g, "Z", 10.50, 106.50)
    with pytest.raises(NoPathFoundError):
        find_path(g, "A", "Z", departure_hour=8.0, corridor_bias=NO_BIAS)


def test_edge_missing_length_raises_instead_of_treating_it_as_free():
    """A `length_m = edge.get("length", 0.0)` default would silently treat
    a malformed edge as instantaneous/zero-cost, biasing the search toward
    it regardless of true distance or congestion."""
    g = nx.MultiDiGraph()
    _add_node(g, "A", 10.80, 106.70)
    _add_node(g, "B", 10.79, 106.70)
    g.add_edge("A", "B", highway="residential", name=None)  # no `length`
    with pytest.raises(MissingEdgeLengthError):
        find_path(g, "A", "B", departure_hour=8.0, corridor_bias=NO_BIAS)


def test_path_cost_matches_manually_summed_edge_costs():
    g = _linear_graph()
    result = find_path(g, "A", "C", departure_hour=8.0, corridor_bias=NO_BIAS)

    t = 8.0
    total = 0.0
    for u, v in [("A", "B"), ("B", "C")]:
        edge = g.get_edge_data(u, v)[0]
        cost = traffic_model.edge_travel_time_hours(edge["length"], edge["highway"], edge["name"], t, NO_BIAS)
        total += cost
        t += cost

    assert result.total_travel_time_hours == pytest.approx(total, rel=1e-6)


def test_time_dependence_changes_edge_speed_along_route():
    """An 8am departure hitting a later edge past 9am should reflect the
    easing-congestion speed at that later time, not the 8am speed.

    Numeric check: at 8.0h, m_time=0.55 (flat 7-9h trough), so the A->B leg
    (33 km at 50 km/h free-flow -> v_eff=27.5 km/h) takes 1.2h, arriving at
    B at 9.2h - inside the 9-10h rising window, where m_time(9.2)=0.59, so
    v_eff there is 50*0.59=29.5 km/h, strictly faster than the first leg."""
    g = nx.MultiDiGraph()
    _add_node(g, "A", 10.90, 106.70)
    _add_node(g, "B", 10.80, 106.70)  # 33 km leg: an 8am departure arrives at B past 9.0h
    _add_node(g, "C", 10.79, 106.70)
    _add_edge(g, "A", "B", 33_000.0, highway="primary")
    _add_edge(g, "B", "C", 500.0, highway="primary")

    result = find_path(g, "A", "C", departure_hour=8.0, corridor_bias=NO_BIAS)
    assert result.edge_trace[1][2] > result.edge_trace[0][2]  # second edge's speed strictly exceeds the first's


def test_greedy_shortest_hop_path_differs_from_true_fastest_path():
    """Two routes of different length/class: the shorter-distance route is
    on the slower road class, so the time-weighted optimum differs from a
    naive shortest-distance choice.

    Numeric check: at any shared departure time, primary (50 km/h) is 2x
    residential's (25 km/h) free-flow speed. Route B1 (residential) totals
    1800 m -> 0.1309h; route B2 (primary) totals 2800 m -> 0.1018h. Despite
    being 1000 m longer, B2 is faster because its distance ratio (2800/1800
    = 1.56) is less than its speed ratio (2x)."""
    g = nx.MultiDiGraph()
    _add_node(g, "A", 10.80, 106.70)
    _add_node(g, "B1", 10.79, 106.705)  # shorter-distance residential detour
    _add_node(g, "B2", 10.79, 106.695)  # longer-distance but faster-class primary detour
    _add_node(g, "C", 10.78, 106.70)
    _add_edge(g, "A", "B1", 900.0, highway="residential")  # 25 km/h free-flow
    _add_edge(g, "B1", "C", 900.0, highway="residential")
    _add_edge(g, "A", "B2", 1400.0, highway="primary")     # 50 km/h free-flow
    _add_edge(g, "B2", "C", 1400.0, highway="primary")

    result = find_path(g, "A", "C", departure_hour=8.0, corridor_bias=NO_BIAS)  # rush hour
    assert result.node_path == ["A", "B2", "C"]  # longer-but-faster road wins


def test_correct_despite_crossing_a_recovery_window():
    """A graph where the optimal route's second leg crosses a recovery
    window (heuristic locally inconsistent there, per test_heuristic.py) and
    a much longer detour exists; correctness (via reopening) must still hold."""
    g = nx.MultiDiGraph()
    _add_node(g, "A", 10.90, 106.70)
    _add_node(g, "B", 10.80, 106.70)
    _add_node(g, "C", 10.79, 106.70)
    _add_node(g, "D", 10.90, 106.71)
    _add_edge(g, "A", "B", 11_000.0, highway="motorway")
    _add_edge(g, "B", "C", 500.0, highway="motorway")
    _add_edge(g, "A", "D", 200.0, highway="residential")
    _add_edge(g, "D", "C", 50_000.0, highway="residential")  # much longer detour

    result = find_path(g, "A", "C", departure_hour=9.0, corridor_bias=NO_BIAS)
    assert result.node_path == ["A", "B", "C"]


def test_genuine_reopening_after_expansion_beats_premature_pop():
    """Forces the case reopening exists for: a node is popped and expanded,
    then reached again by another route with a strictly better arrival time,
    so it must be expanded a second time.

    That only happens where the heuristic is inconsistent, hence departure
    at 16.65h, inside the evening window covered by
    test_heuristic.py::test_consistency_can_fail_somewhere_in_the_day.
    Elsewhere admissibility plus consistency make the first pop of a node
    final.

    Layout (A, X and Goal are collinear on one meridian):
    - A -> X -> Goal is the direct motorway route. Consistency fails on
      A -> X, so f(X) comes out lower than f(A) and X is popped and
      expanded before the better route is even considered.
    - Y sits at A's coordinates (so it has A's heuristic) and is reached by
      a near-instant 10 m edge. Y -> X is a 100 m motorway shortcut, so
      A -> Y -> X reaches X sooner than A -> X does.
    - Without reopening the search returns the direct route (about
      0.2405h). With it, A -> Y -> X -> Goal wins (about 0.1923h).
    """
    point_a = (10.80, 106.70)
    point_x = (10.78, 106.70)
    point_goal = (10.70, 106.70)
    dist_ax_km = heuristic.haversine_km(*point_a, *point_x)
    dist_x_goal_km = heuristic.haversine_km(*point_x, *point_goal)

    g = nx.MultiDiGraph()
    _add_node(g, "A", *point_a)
    _add_node(g, "X", *point_x)
    _add_node(g, "Goal", *point_goal)
    _add_node(g, "Y", *point_a)  # co-located with A
    _add_edge(g, "A", "X", dist_ax_km * 1000.0, highway="motorway")
    _add_edge(g, "X", "Goal", dist_x_goal_km * 1000.0, highway="motorway")
    _add_edge(g, "A", "Y", 10.0, highway="residential")
    _add_edge(g, "Y", "X", 100.0, highway="motorway")

    result = find_path(g, "A", "Goal", departure_hour=16.65, v_free_bound_kmh=80.0, corridor_bias=NO_BIAS)

    # The reopened detour is the one actually returned.
    assert result.node_path == ["A", "Y", "X", "Goal"]

    # It is strictly better than the direct route a non-reopening search
    # would have locked in on X's first (premature) expansion.
    direct_cost_ax = traffic_model.edge_travel_time_hours(dist_ax_km * 1000.0, "motorway", None, 16.65, NO_BIAS)
    direct_arrival_x = 16.65 + direct_cost_ax
    direct_cost_x_goal = traffic_model.edge_travel_time_hours(
        dist_x_goal_km * 1000.0, "motorway", None, direct_arrival_x, NO_BIAS)
    direct_total = direct_cost_ax + direct_cost_x_goal
    assert result.total_travel_time_hours < direct_total - 1e-9

    # And it matches the manually-summed cost of the actual returned path.
    t = 16.65
    total = 0.0
    for u, v in [("A", "Y"), ("Y", "X"), ("X", "Goal")]:
        edge = g.get_edge_data(u, v)[0]
        cost = traffic_model.edge_travel_time_hours(edge["length"], edge["highway"], edge["name"], t, NO_BIAS)
        total += cost
        t += cost
    assert result.total_travel_time_hours == pytest.approx(total, rel=1e-6)


def test_start_equal_to_goal_is_a_zero_cost_path():
    g = _linear_graph()
    result = find_path(g, "B", "B", departure_hour=8.0, corridor_bias=NO_BIAS)
    assert result.node_path == ["B"]
    assert result.total_travel_time_hours == 0.0
    assert result.total_distance_km == 0.0


def test_shortest_parallel_edge_picks_the_shortest_of_several_edges():
    """OSMnx can keep several edges between the same two nodes. The search
    and the map must agree on which one they use."""
    g = nx.MultiDiGraph()
    _add_node(g, "A", 10.80, 106.70)
    _add_node(g, "B", 10.79, 106.70)
    _add_edge(g, "A", "B", 1500.0, highway="residential")
    _add_edge(g, "A", "B", 1100.0, highway="primary")

    assert shortest_parallel_edge(g, "A", "B")["length"] == 1100.0
    result = find_path(g, "A", "B", departure_hour=2.0, corridor_bias=NO_BIAS)
    assert result.total_distance_km == pytest.approx(1.1)
