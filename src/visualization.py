"""Folium map rendering for POIs, road network, and computed paths."""
import os

import folium
import folium.plugins
import pandas as pd

from . import traffic_model
from .astar import _edge_data

CATEGORY_COLORS = {
    "hospital": "red",
    "school": "blue",
    "marketplace": "orange",
    "place_of_worship": "purple",
    "mall": "pink",
    "attraction": "green",
    "museum": "darkgreen",
    "bus_station": "cadetblue",
    "park": "lightgreen",
    "hotel": "navy",
    "supermarket": "teal",
}
DEFAULT_CATEGORY_COLOR = "gray"

# (speed ratio v_eff/v_free upper bound, color) - congested to free-flowing.
SPEED_COLOR_STOPS = [
    (0.35, "#d73027"),
    (0.7, "#fee08b"),
    (1.01, "#1a9850"),
]


def _speed_color(v_eff_kmh, v_free_kmh):
    ratio = 0.0 if v_free_kmh <= 0 else max(0.0, min(1.0, v_eff_kmh / v_free_kmh))
    for stop, color in SPEED_COLOR_STOPS:
        if ratio <= stop:
            return color
    return SPEED_COLOR_STOPS[-1][1]


def _edge_free_flow_speed_kmh(graph, u, v):
    """Free-flow speed of the actual (shortest-of-parallel) edge (u, v) in
    `graph`, by its own OSM road class - reuses astar's own parallel-edge
    selection so the color matches the exact edge the search costed."""
    edge = _edge_data(graph, u, v)
    return traffic_model.free_flow_speed_kmh(edge.get("highway"))


def render_map(graph, pois, path_result=None,
               departure_hour=None, output_path="data/last_result_map.html",
               show_network=True):
    """Build a folium.Map centered on the graph, with POIs, the road
    network, and (if given) a computed path drawn on top. Saves to
    `output_path` and also returns the Map object for inline notebook
    display.

    `show_network=False` skips the thin grey full-road-network context
    lines (POIs, path, markers and legend are unaffected). The HCMC graph
    has ~18.5k edges, so one line per edge dominates the rendered HTML -
    roughly 13.6 MB per map. For a worked example the path plus markers
    plus nearby POIs already tell the story, so the notebook's baked-in
    demo maps turn this off to keep the saved .ipynb a sane size.
    """
    lats = [data["y"] for _, data in graph.nodes(data=True)]
    lons = [data["x"] for _, data in graph.nodes(data=True)]
    center = (sum(lats) / len(lats), sum(lons) / len(lons))

    # "OpenStreetMap" (folium's default) needs no API key, unlike
    # "cartodbpositron" - CartoDB started requiring one, which raised a
    # UserWarning on every render even though the tiles still loaded.
    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    if show_network:
        for u, v, _ in graph.edges(data=True):
            u_data, v_data = graph.nodes[u], graph.nodes[v]
            folium.PolyLine(
                [(u_data["y"], u_data["x"]), (v_data["y"], v_data["x"])],
                color="#cccccc", weight=1, opacity=0.6,
            ).add_to(fmap)

    clusters_by_category = {}
    for _, poi in pois.iterrows():
        category = poi.get("category")
        # `or "other"` mishandles pandas' missing-value markers: pd.NA raises
        # on bool(), and a bare float('nan') is truthy so the fallback would
        # never fire, producing a stray cluster literally named "nan".
        if category is None or pd.isna(category):
            category = "other"
        color = CATEGORY_COLORS.get(category, DEFAULT_CATEGORY_COLOR)
        if category not in clusters_by_category:
            clusters_by_category[category] = folium.plugins.MarkerCluster(
                name=category).add_to(fmap)
        centroid = poi.geometry.centroid
        folium.CircleMarker(
            (centroid.y, centroid.x), radius=4, color=color, fill=True,
            fill_opacity=0.9, popup=poi.get("name", "POI"),
        ).add_to(clusters_by_category[category])
    if clusters_by_category:
        folium.LayerControl(collapsed=False).add_to(fmap)

    if path_result is not None:
        node_data = {n: graph.nodes[n] for n in path_result.node_path}
        for u, v, speed_kmh in path_result.edge_trace:
            color = _speed_color(speed_kmh, _edge_free_flow_speed_kmh(graph, u, v))
            folium.PolyLine(
                [(node_data[u]["y"], node_data[u]["x"]), (node_data[v]["y"], node_data[v]["x"])],
                color=color, weight=5, opacity=0.9,
            ).add_to(fmap)

        start, goal = path_result.node_path[0], path_result.node_path[-1]
        folium.Marker((node_data[start]["y"], node_data[start]["x"]),
                      icon=folium.Icon(color="green"), popup="Start").add_to(fmap)
        folium.Marker((node_data[goal]["y"], node_data[goal]["x"]),
                      icon=folium.Icon(color="red"), popup="Goal").add_to(fmap)

        legend_html = (
            f"<b>Distance:</b> {path_result.total_distance_km:.2f} km<br>"
            f"<b>Travel time:</b> {path_result.total_travel_time_hours * 60:.1f} min<br>"
            f"<b>Departure:</b> {departure_hour if departure_hour is not None else 'n/a'}"
        )
        fmap.get_root().html.add_child(folium.Element(
            f'<div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; '
            f'background: white; padding: 10px; border: 1px solid #999;">{legend_html}</div>'
        ))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fmap.save(output_path)
    return fmap
