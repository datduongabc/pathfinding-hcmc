"""Custom time-dependent A* search over the HCMC road graph."""
import heapq
import itertools
import logging

from . import heuristic as heuristic_mod
from . import traffic_model

logger = logging.getLogger(__name__)


class NoPathFoundError(RuntimeError):
    """Raised when start and goal are in different connected components."""


class MissingEdgeLengthError(RuntimeError):
    """Raised when a graph edge has no `length` attribute.

    Every drivable OSMnx edge carries `length`; treating a missing one as
    0.0 would silently make that edge free (instant, zero-cost) instead of
    surfacing the malformed graph data.
    """


class PathResult:
    def __init__(self, node_path, total_travel_time_hours, total_distance_km, edge_trace):
        self.node_path = node_path
        self.total_travel_time_hours = total_travel_time_hours
        self.total_distance_km = total_distance_km
        self.edge_trace = edge_trace  # list of (u, v, speed_kmh)


def _edge_data(graph, u, v):
    # osmnx MultiDiGraph: take the shortest of any parallel edges between u, v.
    edges = graph.get_edge_data(u, v)
    key = min(edges, key=lambda k: edges[k].get("length", float("inf")))
    return edges[key]


def find_path(graph, start_node, goal_node, departure_hour, v_free_bound_kmh=None,
              corridor_bias=None):
    """Time-dependent A* from `start_node` to `goal_node`, departing at
    `departure_hour` (float, 0-24). Returns a PathResult.

    `graph` is an OSMnx/networkx MultiDiGraph with `x`/`y` node attributes
    (lon/lat) and `highway`/`length`/`name` edge attributes.
    """
    if v_free_bound_kmh is None:
        v_free_bound_kmh = heuristic_mod.compute_free_flow_bound_kmh(graph)
    if corridor_bias is None:
        corridor_bias = traffic_model.load_corridor_bias()

    def node_latlon(n):
        data = graph.nodes[n]
        return data["y"], data["x"]

    goal_lat, goal_lon = node_latlon(goal_node)  # invariant for the whole search

    def h(n, hour):
        lat, lon = node_latlon(n)
        return heuristic_mod.heuristic_hours(lat, lon, goal_lat, goal_lon, hour, v_free_bound_kmh)

    counter = itertools.count()  # tie-breaker so heap never compares node ids
    open_heap = [(h(start_node, departure_hour), next(counter), start_node, departure_hour)]
    best_g = {start_node: departure_hour}
    came_from = {}  # node -> (prev_node, edge_data, speed_kmh)
    edge_cache = {}  # (u, v) -> edge_data, since reopening can revisit the same edge

    while open_heap:
        f, _, node, arrival_hour = heapq.heappop(open_heap)

        if arrival_hour > best_g.get(node, float("inf")) + 1e-9:
            continue  # stale heap entry
        if node == goal_node:
            return _reconstruct(came_from, start_node, goal_node, departure_hour, arrival_hour)

        for neighbor in graph.successors(node):
            edge = edge_cache.get((node, neighbor))
            if edge is None:
                edge = edge_cache[(node, neighbor)] = _edge_data(graph, node, neighbor)
            length_m = edge.get("length")
            if length_m is None:
                raise MissingEdgeLengthError(
                    f"Edge ({node}, {neighbor}) has no 'length' attribute")
            street_name = edge.get("name")
            if isinstance(street_name, list):
                street_name = street_name[0] if street_name else None
            travel_time = traffic_model.edge_travel_time_hours(
                length_m, edge.get("highway"), street_name, arrival_hour, corridor_bias)
            neighbor_arrival = arrival_hour + travel_time

            if neighbor_arrival < best_g.get(neighbor, float("inf")) - 1e-9:
                best_g[neighbor] = neighbor_arrival
                speed_kmh = (length_m / 1000.0) / travel_time if travel_time > 0 else 0.0
                came_from[neighbor] = (node, edge, speed_kmh)
                f_neighbor = neighbor_arrival + h(neighbor, neighbor_arrival)
                heapq.heappush(open_heap, (f_neighbor, next(counter), neighbor, neighbor_arrival))
                # Reopening is deliberate: a node already relaxed can still be
                # re-pushed here on a strictly better g, since the heuristic
                # is only proven consistent outside the rush-hour recovery
                # windows (see docs/superpowers/specs/2026-09-16-hcmc-pathfinding-design.md).

    raise NoPathFoundError(f"No path found from {start_node} to {goal_node}")


def _reconstruct(came_from, start_node, goal_node, departure_hour, goal_arrival_hour):
    path = [goal_node]
    trace = []
    node = goal_node
    total_distance_km = 0.0
    while node != start_node:
        prev_node, edge, speed_kmh = came_from[node]
        trace.append((prev_node, node, speed_kmh))
        # No default here: every edge in came_from already passed the
        # strict `length is not None` check in the main loop above.
        total_distance_km += edge["length"] / 1000.0
        path.append(prev_node)
        node = prev_node
    path.reverse()
    trace.reverse()

    total_travel_time_hours = goal_arrival_hour - departure_hour
    if total_travel_time_hours > traffic_model.WINDOW_HOURS:
        logger.warning(
            "Path travel time %.2fh exceeds heuristic window W=%.1fh; "
            "admissibility proof's assumption may not hold for this result",
            total_travel_time_hours, traffic_model.WINDOW_HOURS)

    return PathResult(path, total_travel_time_hours, total_distance_km, trace)
