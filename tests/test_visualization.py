import folium
import folium.plugins
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from src import data_collection, visualization
from src.astar import PathResult

NETWORK_LINE_COLOR = "#cccccc"
NETWORK_LINE_WEIGHT = 1
PATH_LINE_WEIGHT = 5


def _tiny_graph():
    g = nx.MultiDiGraph()
    g.add_node("A", y=10.80, x=106.70)
    g.add_node("B", y=10.79, x=106.70)
    g.add_edge("A", "B", length=1000.0, highway="residential", name="Test St")
    return g


def _tiny_pois():
    """POIs shaped like the real cached file: raw OSM tag columns with the
    `category` column derived from them, not hardcoded."""
    raw = gpd.GeoDataFrame(
        [{"name": "Test POI", "amenity": "school", "geometry": Point(106.70, 10.80)}],
        crs="EPSG:4326",
    )
    return data_collection.derive_category(raw)


def _path_result(speed_kmh):
    return PathResult(["A", "B"], total_travel_time_hours=0.05,
                      total_distance_km=1.0, edge_trace=[("A", "B", speed_kmh)])


# The tests below inspect the folium objects the map is built from instead
# of searching the rendered HTML, so they don't break when folium changes
# how it writes markup. `_children` is folium/branca's own child registry;
# it is reached only through this one helper.
def _elements_of_type(fmap, element_type):
    found = []

    def walk(element):
        for child in element._children.values():
            if isinstance(child, element_type):
                found.append(child)
            walk(child)

    walk(fmap)
    return found


def _path_line_colors(fmap):
    return [line.options["color"] for line in _elements_of_type(fmap, folium.PolyLine)
            if line.options["weight"] == PATH_LINE_WEIGHT]


def _network_lines(fmap):
    return [line for line in _elements_of_type(fmap, folium.PolyLine)
            if line.options["weight"] == NETWORK_LINE_WEIGHT
            and line.options["color"] == NETWORK_LINE_COLOR]


def _cluster_names(fmap):
    return {cluster.layer_name for cluster in _elements_of_type(fmap, folium.plugins.MarkerCluster)}


def test_category_colors_cover_every_enumerable_poi_tag_value():
    """CATEGORY_COLORS and POI_TAGS are maintained by hand separately, and a
    POI category with no color silently renders grey. Only enumerable tag
    values are checked: an open-ended key like `historic: True` accepts any
    value, so those fall back to DEFAULT_CATEGORY_COLOR by design."""
    enumerable_values = {
        value
        for values in data_collection.POI_TAGS.values()
        if values is not True
        for value in values
    }
    missing = enumerable_values - set(visualization.CATEGORY_COLORS)
    assert not missing, f"POI_TAGS values with no CATEGORY_COLORS entry: {missing}"


@pytest.mark.parametrize("v_eff, v_free, expected_color", [
    (5.0, 25.0, visualization.SPEED_COLOR_STOPS[0][1]),    # 20% of free-flow: congested
    (15.0, 25.0, visualization.SPEED_COLOR_STOPS[1][1]),   # 60%: slow
    (25.0, 25.0, visualization.SPEED_COLOR_STOPS[2][1]),   # 100%: free-flowing
    (40.0, 25.0, visualization.SPEED_COLOR_STOPS[2][1]),   # above free-flow is capped
    (10.0, 0.0, visualization.SPEED_COLOR_STOPS[0][1]),    # no free-flow speed: treat as congested
])
def test_speed_color_reflects_the_share_of_free_flow_speed(v_eff, v_free, expected_color):
    assert visualization.speed_color(v_eff, v_free) == expected_color


def test_render_map_without_path_produces_html(tmp_path):
    output = tmp_path / "map.html"
    fmap = visualization.render_map(_tiny_graph(), _tiny_pois(), output_path=str(output))
    assert output.exists()
    assert output.stat().st_size > 0
    assert fmap is not None


def test_render_map_with_path_includes_legend(tmp_path):
    output = tmp_path / "map.html"
    visualization.render_map(_tiny_graph(), _tiny_pois(), path_result=_path_result(20.0),
                             departure_hour=8.0, output_path=str(output))
    # The legend is user-visible text, so its content is checked as rendered.
    html = output.read_text(encoding="utf-8")
    assert "Travel time" in html
    assert "3.0 min" in html


def test_render_map_colors_markers_by_derived_category(tmp_path):
    """The derived category must reach the marker's color, not fall through
    to the grey default."""
    fmap = visualization.render_map(_tiny_graph(), _tiny_pois(),
                                    output_path=str(tmp_path / "map.html"))
    marker_colors = {m.options["color"] for m in _elements_of_type(fmap, folium.CircleMarker)}
    assert marker_colors == {visualization.CATEGORY_COLORS["school"]}
    assert _cluster_names(fmap) == {"school"}


def test_render_map_without_network_omits_context_lines_but_keeps_the_path(tmp_path):
    """show_network=False drops the full-road-network context lines, which
    dominate the HTML size, while keeping the path, its markers and the
    legend."""
    graph = _tiny_graph()
    result = _path_result(20.0)

    with_network = visualization.render_map(
        graph, _tiny_pois(), path_result=result, departure_hour=8.0,
        output_path=str(tmp_path / "with.html"), show_network=True)
    without_network = visualization.render_map(
        graph, _tiny_pois(), path_result=result, departure_hour=8.0,
        output_path=str(tmp_path / "without.html"), show_network=False)

    assert len(_network_lines(with_network)) == graph.number_of_edges()
    assert _network_lines(without_network) == []
    assert (tmp_path / "without.html").stat().st_size < (tmp_path / "with.html").stat().st_size

    # Path, markers and legend survive. The tiny graph's A-B edge is
    # "residential" (25 km/h free-flow, see traffic_model.FREE_FLOW_SPEED_KMH),
    # so 20 km/h is 80% of its own free-flow speed.
    assert _path_line_colors(without_network) == [visualization.speed_color(20.0, 25.0)]
    assert len(_elements_of_type(without_network, folium.Marker)) >= 2  # start and goal
    assert len(_elements_of_type(without_network, folium.CircleMarker)) == 1  # the POI
    html = (tmp_path / "without.html").read_text(encoding="utf-8")
    assert "Travel time" in html
    assert "Test POI" in html


def test_render_map_defaults_to_showing_the_network(tmp_path):
    graph = _tiny_graph()
    fmap = visualization.render_map(graph, _tiny_pois(), output_path=str(tmp_path / "map.html"))
    assert len(_network_lines(fmap)) == graph.number_of_edges()


def test_render_map_colors_path_by_the_edges_own_free_flow_speed(tmp_path):
    """A residential edge (25 km/h free-flow) running at its own full
    free-flow speed must render green. Comparing against one network-wide
    bound (80 km/h, motorway-class) would show it red, as if congested."""
    fmap = visualization.render_map(_tiny_graph(), _tiny_pois(), path_result=_path_result(25.0),
                                    departure_hour=8.0, output_path=str(tmp_path / "map.html"))
    green = visualization.SPEED_COLOR_STOPS[-1][1]
    assert _path_line_colors(fmap) == [green]
    assert visualization.speed_color(25.0, 80.0) != green  # what a network-wide bound would give


def test_render_map_handles_pd_na_category_without_crashing(tmp_path):
    """derive_category leaves a row's category as pd.NA when none of
    POI_TAGS' columns are present. bool(pd.NA) is ambiguous and raises, so
    the renderer must not test it for truthiness."""
    raw = gpd.GeoDataFrame(
        [{"name": "No Tags POI", "geometry": Point(106.70, 10.80)}],
        crs="EPSG:4326",
    )
    pois = data_collection.derive_category(raw)
    assert pois["category"].isna().all()

    fmap = visualization.render_map(_tiny_graph(), pois, output_path=str(tmp_path / "map.html"))
    assert _cluster_names(fmap) == {"other"}
    marker_colors = {m.options["color"] for m in _elements_of_type(fmap, folium.CircleMarker)}
    assert marker_colors == {visualization.DEFAULT_CATEGORY_COLOR}


def test_render_map_handles_float_nan_category_without_a_stray_nan_cluster(tmp_path):
    """A plain float('nan') is truthy, so a truthiness fallback to "other"
    never fires. Each nan would then get its own cluster labeled "nan"."""
    pois = _tiny_pois()
    pois["category"] = pd.Series([np.nan], index=pois.index)

    fmap = visualization.render_map(_tiny_graph(), pois, output_path=str(tmp_path / "map.html"))
    assert _cluster_names(fmap) == {"other"}
