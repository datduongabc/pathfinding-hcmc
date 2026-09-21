import networkx as nx
import pytest

from src.astar import MissingEdgeLengthError, NoPathFoundError, find_path


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
    result = find_path(g, "A", "C", departure_hour=2.0)
    assert result.node_path == ["A", "B", "C"]
    assert result.total_travel_time_hours > 0


def test_disconnected_start_end_raises():
    g = nx.MultiDiGraph()
    _add_node(g, "A", 10.80, 106.70)
    _add_node(g, "Z", 10.50, 106.50)
    with pytest.raises(NoPathFoundError):
        find_path(g, "A", "Z", departure_hour=8.0)


def test_edge_missing_length_raises_instead_of_treating_it_as_free():
    """A `length_m = edge.get("length", 0.0)` default would silently treat
    a malformed edge as instantaneous/zero-cost, biasing the search toward
    it regardless of true distance or congestion."""
    g = nx.MultiDiGraph()
    _add_node(g, "A", 10.80, 106.70)
    _add_node(g, "B", 10.79, 106.70)
    g.add_edge("A", "B", highway="residential", name=None)  # no `length`
    with pytest.raises(MissingEdgeLengthError):
        find_path(g, "A", "B", departure_hour=8.0)


def test_path_cost_matches_manually_summed_edge_costs():
    g = _linear_graph()
    result = find_path(g, "A", "C", departure_hour=8.0)

    from src import traffic_model
    t = 8.0
    total = 0.0
    for u, v in [("A", "B"), ("B", "C")]:
        edge = g.get_edge_data(u, v)[0]
        cost = traffic_model.edge_travel_time_hours(edge["length"], edge["highway"], edge["name"], t, {})
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

    result = find_path(g, "A", "C", departure_hour=8.0)
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

    result = find_path(g, "A", "C", departure_hour=8.0)  # rush hour
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

    result = find_path(g, "A", "C", departure_hour=9.0)
    assert result.node_path == ["A", "B", "C"]


def test_genuine_reopening_after_expansion_beats_premature_pop():
    """Forces the specific scenario reopening exists to handle: a node
    popped and *expanded* (its own successors already relaxed), then later
    rediscovered via a different route with a strictly better g, requiring
    it to be reopened after having already been "closed" once.

    This can only be forced by routing a real edge traversal through one of
    the narrow windows where the windowed heuristic is locally inconsistent
    (see test_heuristic.py's `test_consistency_can_fail_somewhere_in_the_day`
    and its crossing-point derivation) - everywhere else, admissibility +
    consistency together guarantee "first pop is final," so no genuine
    reopening-after-expansion can ever be forced there.

    Numeric derivation (reusing test_heuristic.py's collinear A/B/Goal
    points and the same evening-window crossing hand-verified for Task 4/6,
    t=16.65h, consistency gap +0.0014 on the A->X leg):
      - A=(10.80,106.70), X=(10.78,106.70), Goal=(10.70,106.70), all on the
        same meridian; A->X and X->Goal are motorway (80 km/h free-flow),
        lengths equal to the exact haversine distances between those points.
      - At departure_hour=16.65, cost(A,X)=0.047317h -> X's first arrival
        is 16.697317h, with f(X)=16.881233 - strictly *less* than f(A)
        itself (16.882625, i.e. h(A,16.65)) because the consistency
        inequality is violated on this exact leg. This deceptively low f
        is what causes X to be popped and expanded before the (better)
        detour below is even considered - X's first-round expansion relaxes
        X->Goal, arriving at 16.890474h (f=16.890474, since h(Goal)=0).
      - Y is a second node co-located with A (same lat/lon, so h(Y,*)
        matches h(A,*)), reached by a near-instant 10 m residential edge
        (A->Y, cost~0.0004h). Its own f (~16.883266) lands strictly between
        f(X)=16.881233 and f(Goal-via-X)=16.890474 - Y pops *after* X's
        first expansion but *before* the algorithm can return the
        goal-via-X-direct result.
      - Y->X is a short 100 m motorway shortcut: expanding Y gives a second,
        strictly better arrival at X (16.652809h < X's current best of
        16.697317h) - forcing X to be reopened and re-pushed with a lower f
        (16.838778), even though X had already been popped and expanded
        once. Re-expanding X then finds a strictly better arrival at Goal
        (16.842304h) than the stale, already-open direct-route entry
        (16.890474h), which is what the algorithm ultimately returns.
      - Without reopening (a permanent closed set blocking re-relaxation of
        an already-expanded X), the search would return the direct route
        A->X->Goal (total ~0.240474h) instead - strictly worse than the
        actual optimum A->Y->X->Goal (~0.192304h) that reopening finds.

    (Derived and iteratively confirmed via temporary pop-order
    instrumentation during development; the shipped test below asserts only
    on find_path's return value.)
    """
    from src import heuristic, traffic_model

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

    result = find_path(g, "A", "Goal", departure_hour=16.65, v_free_bound_kmh=80.0, corridor_bias={})

    # The reopened detour is the one actually returned.
    assert result.node_path == ["A", "Y", "X", "Goal"]

    # It is strictly better than the direct route a non-reopening search
    # would have locked in on X's first (premature) expansion.
    direct_cost_ax = traffic_model.edge_travel_time_hours(dist_ax_km * 1000.0, "motorway", None, 16.65, {})
    direct_arrival_x = 16.65 + direct_cost_ax
    direct_cost_x_goal = traffic_model.edge_travel_time_hours(
        dist_x_goal_km * 1000.0, "motorway", None, direct_arrival_x, {})
    direct_total = direct_cost_ax + direct_cost_x_goal
    assert result.total_travel_time_hours < direct_total - 1e-9

    # And it matches the manually-summed cost of the actual returned path.
    t = 16.65
    total = 0.0
    for u, v in [("A", "Y"), ("Y", "X"), ("X", "Goal")]:
        edge = g.get_edge_data(u, v)[0]
        cost = traffic_model.edge_travel_time_hours(edge["length"], edge["highway"], edge["name"], t, {})
        total += cost
        t += cost
    assert result.total_travel_time_hours == pytest.approx(total, rel=1e-6)
