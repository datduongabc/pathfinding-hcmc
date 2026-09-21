import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src import data_collection, visualization
from src.astar import PathResult


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


def test_render_map_without_path_produces_html(tmp_path):
    output = tmp_path / "map.html"
    fmap = visualization.render_map(_tiny_graph(), _tiny_pois(), output_path=str(output))
    assert output.exists()
    assert output.stat().st_size > 0
    assert fmap is not None


def test_render_map_with_path_includes_legend(tmp_path):
    output = tmp_path / "map.html"
    result = PathResult(["A", "B"], total_travel_time_hours=0.05,
                         total_distance_km=1.0, edge_trace=[("A", "B", 20.0)])
    visualization.render_map(_tiny_graph(), _tiny_pois(), path_result=result,
                              departure_hour=8.0, output_path=str(output))
    html = output.read_text(encoding="utf-8")
    assert "Travel time" in html


def test_render_map_colors_markers_by_derived_category(tmp_path):
    """The derived category must reach the marker's color, not fall through
    to the grey default."""
    output = tmp_path / "map.html"
    visualization.render_map(_tiny_graph(), _tiny_pois(), output_path=str(output))
    html = output.read_text(encoding="utf-8")
    assert visualization.CATEGORY_COLORS["school"] in html
    assert visualization.DEFAULT_CATEGORY_COLOR not in html


def test_render_map_without_network_omits_context_lines_but_keeps_the_path(tmp_path):
    """show_network=False drops the full-road-network context lines, which
    dominate the HTML size, while keeping the path, its markers and the
    legend."""
    graph = _tiny_graph()
    result = PathResult(["A", "B"], total_travel_time_hours=0.05,
                        total_distance_km=1.0, edge_trace=[("A", "B", 20.0)])

    with_network = tmp_path / "with.html"
    visualization.render_map(graph, _tiny_pois(), path_result=result,
                             departure_hour=8.0, output_path=str(with_network),
                             show_network=True)
    without_network = tmp_path / "without.html"
    visualization.render_map(graph, _tiny_pois(), path_result=result,
                             departure_hour=8.0, output_path=str(without_network),
                             show_network=False)

    with_html = with_network.read_text(encoding="utf-8")
    without_html = without_network.read_text(encoding="utf-8")

    assert "#cccccc" in with_html, "network context lines missing when show_network=True"
    assert "#cccccc" not in without_html, "network context lines drawn when show_network=False"
    assert len(without_html) < len(with_html)

    # Path, markers and legend survive. The tiny graph's A-B edge is
    # "residential" (25 km/h free-flow, see traffic_model.FREE_FLOW_SPEED_KMH),
    # so 20 km/h is 80% of its own free-flow speed.
    assert "Travel time" in without_html
    assert "Test POI" in without_html
    assert visualization._speed_color(20.0, 25.0) in without_html


def test_render_map_defaults_to_showing_the_network(tmp_path):
    output = tmp_path / "map.html"
    visualization.render_map(_tiny_graph(), _tiny_pois(), output_path=str(output))
    assert "#cccccc" in output.read_text(encoding="utf-8")


def test_render_map_colors_path_by_the_edges_own_free_flow_speed(tmp_path):
    """A residential edge (25 km/h free-flow) running at its own full
    free-flow speed must render green. Comparing against one network-wide
    bound (80 km/h, motorway-class) would show it red, as if congested."""
    output = tmp_path / "map.html"
    result = PathResult(["A", "B"], total_travel_time_hours=0.04,
                         total_distance_km=1.0, edge_trace=[("A", "B", 25.0)])
    visualization.render_map(_tiny_graph(), _tiny_pois(), path_result=result,
                              departure_hour=8.0, output_path=str(output))
    html = output.read_text(encoding="utf-8")
    assert visualization._speed_color(25.0, 25.0) in html  # green: 100% of its own free-flow
    assert visualization._speed_color(25.0, 80.0) not in html  # what a network-wide bound would give


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

    output = tmp_path / "map.html"
    fmap = visualization.render_map(_tiny_graph(), pois, output_path=str(output))
    assert fmap is not None
    assert "other" in output.read_text(encoding="utf-8")


def test_render_map_handles_float_nan_category_without_a_stray_nan_cluster(tmp_path):
    """A plain float('nan') is truthy, so a truthiness fallback to "other"
    never fires. Each nan would then get its own cluster labeled "nan"."""
    pois = _tiny_pois()
    pois["category"] = pd.Series([np.nan], index=pois.index)

    output = tmp_path / "map.html"
    visualization.render_map(_tiny_graph(), pois, output_path=str(output))
    html = output.read_text(encoding="utf-8")
    assert '"other"' in html or ">other<" in html
    assert '"nan"' not in html and ">nan<" not in html
