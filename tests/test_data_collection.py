import os

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import Point, Polygon

from src import data_collection

# The raw OSM tag keys/values the POI query asks for. `category` must be
# derived from these, never hardcoded in a fixture: a hardcoded category
# hides a missing derivation step.
LISTED_TAG_VALUES = {
    value
    for values in data_collection.POI_TAGS.values()
    if values is not True
    for value in values
}


def _fake_pois(n, start_lon=106.69, start_lat=10.77):
    """Raw-OSM-shaped POIs: separate tag columns, no `category` column -
    the same shape ox.features_from_bbox actually returns."""
    rows = []
    for i in range(n):
        rows.append({
            "name": f"POI {i}",
            # Alternate the matched tag key so category derivation has to
            # coalesce across columns rather than read a single one.
            "amenity": "school" if i % 2 == 0 else None,
            "tourism": None if i % 2 == 0 else "museum",
            "geometry": Point(start_lon + i * 0.0001, start_lat + i * 0.0001),
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def test_osmnx_bbox_is_west_south_east_north():
    """osmnx>=2.0 expects (left, bottom, right, top) = (west, south, east,
    north). A wrong order does not raise, it just downloads the wrong area,
    so the order is pinned here against BBOX."""
    assert data_collection.OSMNX_BBOX == (
        data_collection.BBOX["west"], data_collection.BBOX["south"],
        data_collection.BBOX["east"], data_collection.BBOX["north"],
    )


def test_sample_pois_returns_exactly_target_count():
    raw = _fake_pois(200)
    sampled = data_collection.sample_pois(raw, target_count=150, seed=42)
    assert len(sampled) == 150


def test_sample_pois_is_deterministic_with_fixed_seed():
    raw = _fake_pois(200)
    first = data_collection.sample_pois(raw, target_count=150, seed=42)
    second = data_collection.sample_pois(raw, target_count=150, seed=42)
    assert list(first["name"]) == list(second["name"])


def test_sample_pois_raises_when_too_few():
    raw = _fake_pois(80)
    with pytest.raises(data_collection.InsufficientPOIsError):
        data_collection.sample_pois(raw, target_count=150, seed=42)


def test_sample_pois_filters_out_unnamed_entries_before_counting():
    raw = _fake_pois(160)
    raw.loc[0:20, "name"] = None  # 21 unnamed rows -> 139 named, below target
    with pytest.raises(data_collection.InsufficientPOIsError):
        data_collection.sample_pois(raw, target_count=150, seed=42)


def test_sample_pois_returns_exactly_target_when_named_count_equals_target():
    raw = _fake_pois(150)
    sampled = data_collection.sample_pois(raw, target_count=150, seed=42)
    assert len(sampled) == 150
    assert set(sampled["name"]) == set(raw["name"])


def test_sample_pois_derives_category_from_raw_tag_columns():
    raw = _fake_pois(160)
    sampled = data_collection.sample_pois(raw, target_count=150, seed=42)
    assert sampled["category"].notna().all()
    assert set(sampled["category"]) == {"school", "museum"}


def test_derive_category_prefers_the_tag_that_actually_matched_the_query():
    # A theatre tagged tourism=attraction is returned by the query because
    # of `tourism`, not `amenity=theatre` (which the query never asked
    # for), so its category must be "attraction".
    raw = gpd.GeoDataFrame(
        [{"name": "Opera House", "amenity": "theatre", "tourism": "attraction",
          "geometry": Point(106.70, 10.78)}],
        crs="EPSG:4326",
    )
    assert data_collection.derive_category(raw)["category"].iloc[0] == "attraction"


def test_derive_category_accepts_any_value_for_open_ended_tag_keys():
    # POI_TAGS requests `historic: True`, i.e. any historic=* value.
    raw = gpd.GeoDataFrame(
        [{"name": "Independence Palace Tank", "historic": "tank",
          "geometry": Point(106.69, 10.77)}],
        crs="EPSG:4326",
    )
    assert data_collection.derive_category(raw)["category"].iloc[0] == "tank"


@pytest.mark.skipif(not os.path.exists(data_collection.POIS_PATH),
                    reason="cached POI file not present")
def test_committed_pois_file_has_a_real_category_for_every_row():
    """Guards the shipped data, not just the code: every row of the committed
    data/pois.geojson needs a category that visualization.py can color and
    cluster by, or all markers collapse into one grey bucket."""
    pois = gpd.read_file(data_collection.POIS_PATH)

    assert len(pois) == data_collection.TARGET_POI_COUNT
    assert "category" in pois.columns, "committed POI file has no category column"
    assert pois["category"].notna().all(), "some committed POIs have no category"

    for _, row in pois.iterrows():
        category = row["category"]
        # Either one of the explicitly requested tag values, or - for the
        # open-ended `historic: True` key - that row's own historic value.
        assert category in LISTED_TAG_VALUES or category == row.get("historic"), (
            f"POI {row['name']!r} has unexpected category {category!r}"
        )

    # Real data spans many categories; a handful proves they are not all
    # collapsed into one.
    assert len(set(pois["category"])) >= 5


def _two_node_graph():
    # osmnx's nearest_nodes needs the graph's CRS, which real downloaded
    # graphs always carry.
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node("SW", y=10.770, x=106.690)
    g.add_node("NE", y=10.780, x=106.700)
    return g


def test_snap_pois_to_graph_picks_the_nearest_node():
    pois = gpd.GeoDataFrame(
        [{"name": "Near SW", "geometry": Point(106.6905, 10.7702)},
         {"name": "Near NE", "geometry": Point(106.6995, 10.7798)}],
        crs="EPSG:4326",
    )
    snapped = data_collection.snap_pois_to_graph(pois, _two_node_graph())
    assert list(snapped["node_id"]) == ["SW", "NE"]


def test_snap_pois_to_graph_uses_the_centroid_of_a_polygon_poi():
    # A park-sized polygon centered on the NE node: the snap follows its
    # centroid (an area-wide POI still has one entrance point in the graph).
    park = Polygon([(106.6985, 10.7785), (106.7015, 10.7785),
                    (106.7015, 10.7815), (106.6985, 10.7815)])
    pois = gpd.GeoDataFrame([{"name": "Park", "geometry": park}], crs="EPSG:4326")
    snapped = data_collection.snap_pois_to_graph(pois, _two_node_graph())
    assert snapped["node_id"].iloc[0] == "NE"
    assert "node_id" not in pois.columns  # the input frame is left untouched


def test_load_or_download_graph_reports_a_download_failure(tmp_path, monkeypatch):
    def offline(**kwargs):
        raise OSError("no network")

    monkeypatch.setattr(data_collection.ox, "graph_from_bbox", offline)
    with pytest.raises(RuntimeError, match="no cache"):
        data_collection.load_or_download_graph(str(tmp_path / "graph.graphml"))
