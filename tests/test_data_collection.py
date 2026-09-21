import os

import geopandas as gpd
import pytest
from shapely.geometry import Point

from src import data_collection

# The raw OSM tag keys/values the POI query asks for; `category` must be
# derived from these, never hardcoded (a hardcoded fixture category is
# exactly what let the missing-category bug ship).
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
    north) - this exact ordering bug has already shipped twice (two
    separate historical fix commits, each correcting one of the two call
    sites and missing the other), so it's locked down here against BBOX
    directly rather than trusting the two call sites to stay in sync by hand."""
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
    """Regression guard for the shipped data, not just the code: every row
    of the committed data/pois.geojson must carry a category that
    visualization.py can color/cluster by, rather than silently falling
    back to a single grey "other" bucket."""
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

    # The bug being guarded against produced exactly one bucket for all
    # 150 markers; real data spans many categories.
    assert len(set(pois["category"])) >= 5
