"""OSM road-network download and POI extraction/caching for central HCMC."""
import logging
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd

logger = logging.getLogger(__name__)

# Central HCMC districts bounding box (District 1, 3, 4, 5, 10, Binh Thanh,
# Phu Nhuan) - north, south, east, west.
BBOX = {"north": 10.8105, "south": 10.7350, "east": 106.7250, "west": 106.6550}

# osmnx>=2.0 expects bbox as (left, bottom, right, top) = (west, south, east,
# north) - built once here so the two download call sites below can't drift
# apart the way they already have (two separate historical bug-fix commits
# each corrected one call site's argument order and missed the other).
OSMNX_BBOX = (BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"])

POI_TAGS = {
    "amenity": ["hospital", "school", "marketplace", "place_of_worship", "bus_station"],
    "tourism": ["attraction", "museum", "hotel"],
    "shop": ["mall", "supermarket"],
    "leisure": ["park"],
    "historic": True,
}

TARGET_POI_COUNT = 150
SAMPLE_SEED = 42

GRAPH_PATH = "data/graph.graphml"
POIS_PATH = "data/pois.geojson"


class InsufficientPOIsError(RuntimeError):
    """Raised when fewer than TARGET_POI_COUNT named POIs are found."""


def derive_category(pois):
    """Return `pois` with a single `category` column coalesced from the raw
    OSM tag columns.

    `ox.features_from_bbox` returns one column per OSM tag key
    (`amenity`, `tourism`, `shop`, ...), never a unified category, so
    downstream consumers (visualization.py's per-category colors and marker
    clusters) have nothing to group by unless we build it here.

    For each row, the category is the value of the first POI_TAGS column
    (in POI_TAGS order) whose value is one the query actually asked for -
    i.e. the tag that caused the feature to be returned at all. A row can
    carry unrelated extra tags (a `tourism=attraction` theatre also tagged
    `amenity=theatre`); those are not categories and are skipped. Tag keys
    requested with `True` (`historic`) accept any value.
    """
    pois = pois.copy()
    category = pd.Series(pd.NA, index=pois.index, dtype="object")
    for column, wanted_values in POI_TAGS.items():
        if column not in pois.columns:
            continue
        values = pois[column]
        if wanted_values is not True:
            values = values.where(values.isin(wanted_values))
        category = category.fillna(values)
    pois["category"] = category
    return pois


def load_or_download_graph(graph_path=GRAPH_PATH):
    path = Path(graph_path)
    if path.exists():
        return ox.load_graphml(path)
    try:
        graph = ox.graph_from_bbox(
            bbox=OSMNX_BBOX,
            network_type="drive",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download OSM road network and no cache at {path}. "
            "Check network connectivity, or restore the committed cache file."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, path)
    return graph


def sample_pois(raw_pois, target_count=TARGET_POI_COUNT, seed=SAMPLE_SEED):
    """Deterministically sample/trim `raw_pois` (a GeoDataFrame) down to
    exactly `target_count` rows with a name and geometry.

    The result carries a `category` column derived from the raw OSM tag
    columns (see derive_category).

    Raises InsufficientPOIsError if fewer than target_count remain after
    filtering out unnamed/empty-geometry entries - the target is a specific
    count, not just a floor.
    """
    named = raw_pois[raw_pois["name"].notna() & (raw_pois["name"].str.strip() != "")]
    named = named[~named.geometry.is_empty & named.geometry.notna()]
    if len(named) < target_count:
        raise InsufficientPOIsError(
            f"Only {len(named)} named POIs found in bounding box; need {target_count}."
        )
    sampled = named.sample(n=target_count, random_state=seed).reset_index(drop=True)
    return derive_category(sampled)


def load_or_download_pois(pois_path=POIS_PATH, target_count=TARGET_POI_COUNT):
    path = Path(pois_path)
    if path.exists():
        return gpd.read_file(path)
    try:
        raw = ox.features_from_bbox(
            bbox=OSMNX_BBOX,
            tags=POI_TAGS,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download OSM POIs and no cache at {path}. "
            "Check network connectivity, or restore the committed cache file."
        ) from exc
    pois = sample_pois(raw, target_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    pois.to_file(path, driver="GeoJSON")
    return pois


def snap_pois_to_graph(pois, graph):
    """Return `pois` with a `node_id` column, each POI snapped to its
    nearest graph node."""
    xs = pois.geometry.centroid.x
    ys = pois.geometry.centroid.y
    node_ids = ox.distance.nearest_nodes(graph, xs, ys)
    pois = pois.copy()
    pois["node_id"] = node_ids
    return pois
