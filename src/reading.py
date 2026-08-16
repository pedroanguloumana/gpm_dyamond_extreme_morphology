# Functions for loading GPM and DYAMOND data
import numpy as np
import polars as pl
import xarray as xr
from src.gpm_regions import (
    get_region_keys,
    get_region
)
from src.paths import (
    DATA_DIR
)
import os
import pickle
from src.paths import DATA_DIR

# Great-circle km per degree of latitude on a 6371 km sphere
KM_PER_DEG = 6371.0 * np.pi / 180.0

def read_gpm_feature_stats(
        threshold: str,
        res: str,
        maxpr_min,
        maxpr_max,
        edge_frac_max,
        abs_lat_max,
        size_px_min,
):
    
    f = DATA_DIR / "gpm_data" / f"merged_thresh{threshold}_res{res}.csv"
    assert os.path.isfile(f), f"No GPM feature file at {f}"
    df = pl.read_csv(f)
    df = df.with_columns(((pl.col("centroid_lon") + 180) % 360 - 180).alias("centroid_lon"))

    # Cut along region bounds
    _REGION_FROM_SOURCE_FILE = r"_([A-Za-z0-9]+)\.nc$"
    _BOX_SAMPLED_BY: dict[str, str] = {"H02p5": "H02"} # Little bookeeping
    df = df.with_columns(
        pl.col("source_file").str.extract(_REGION_FROM_SOURCE_FILE, 1).alias("region")
    )

    in_own_box = pl.lit(False)
    for key in get_region_keys():
        region = get_region(key)
        
        in_own_box = in_own_box | (
            (pl.col("region") == _BOX_SAMPLED_BY.get(key, key))
            & (pl.col("centroid_lat") >= region.lat_south)
            & (pl.col("centroid_lat") < region.lat_north)
            & (pl.col("centroid_lon") >= region.lon_west)
            & (pl.col("centroid_lon") < region.lon_east)
        )

    # Apply filtering
    df = df.filter(
        in_own_box,
        (pl.col("max_precip_mm_hr") >= maxpr_min)
        & (pl.col("max_precip_mm_hr") < maxpr_max)
        & (pl.col("size_px") >= size_px_min)
        & (pl.col("cross_track_edge_px") / pl.col("perimeter_px") <= edge_frac_max)
        & (pl.col("centroid_lat").abs() <= abs_lat_max)
    )
    return df

# First-generation feature tables only exist for these four models
ARCHIVED_MODELS = {
    "GEOS": "GEOS-3km",
    "ICON": "ICON-SAP-5km",
    "SCREAM": "SCREAM-3km",
    "gSAM": "gSAM-4km",
}

_ARCHIVED_COLUMNS = [
    "source_file",
    "feature_id",
    "is_complete",
    "mean_latitude",
    "mean_longitude",
    "min_latitude",
    "max_latitude",
    "min_longitude",
    "max_longitude",
    "number_pixels",
    "max_precip",
]

def read_archived_dyamond_feature_stats(
    model: str,
    maxpr_min=10.0,
    maxpr_max=np.inf,
    abs_lat_max=20.0,
    size_px_min=5,
    require_complete=True,
    deduplicate=True,
):
    """First-generation DYAMOND feature table (0.05 deg, conservative regrid).

    Before deduplication there is one row per (feature, pseudo-swath): whole-feature
    statistics are repeated on every one of the 149 pseudo-swaths the feature touches.
    The bounding box (``min/max_latitude``, ``min/max_longitude``) is lat/lon-aligned,
    unlike the swath-aligned extents ``read_dyamond_feature_stats`` returns, so
    ``lat_extent_km`` / ``lon_extent_km`` here and ``along/cross_swath_extent_km`` there
    agree only up to feature orientation.
    """
    assert model in ARCHIVED_MODELS, (
        f"No archived features for {model!r}, have {list(ARCHIVED_MODELS)}"
    )
    f = (
        DATA_DIR
        / "old_data"
        / f"merged.{ARCHIVED_MODELS[model]}_0p05deg_conservative_features.csv"
    )
    assert os.path.isfile(f), f"No archived feature file at {f}"

    lf = pl.scan_csv(f).select(_ARCHIVED_COLUMNS)

    keep = (
        (pl.col("max_precip") >= maxpr_min)
        & (pl.col("max_precip") < maxpr_max)
        & (pl.col("number_pixels") >= size_px_min)
        & (pl.col("mean_latitude").abs() <= abs_lat_max)
    )
    if require_complete:
        keep = keep & pl.col("is_complete")
    lf = lf.filter(keep)

    if deduplicate:
        lf = lf.unique(subset=["source_file", "feature_id"])

    df = lf.collect(engine="streaming")

    df = df.with_columns(((pl.col("mean_longitude") + 180) % 360 - 180).alias("mean_longitude"))

    # Boxes straddling the dateline come back as (max - min) close to 360
    lon_span = pl.col("max_longitude") - pl.col("min_longitude")
    lon_span = pl.when(lon_span > 180).then(360 - lon_span).otherwise(lon_span)

    df = df.with_columns(
        # clip() only removes float noise of order 1e-12 km
        (((pl.col("max_latitude") - pl.col("min_latitude")) * KM_PER_DEG)
         .clip(lower_bound=0.0).alias("lat_extent_km")),
        ((lon_span * KM_PER_DEG * pl.col("mean_latitude").radians().cos())
         .clip(lower_bound=0.0).alias("lon_extent_km")),
    )
    return df

def read_dyamond_feature_stats(
    model: str,
    accum: str,
    res: str,
    maxpr_min,
    maxpr_max,
    edge_frac_max,
    abs_lat_max,
    size_px_min,
):
    f = (
        DATA_DIR / "dyamond_data" / model / f"{model}_features_{accum}_{res}deg.csv"
    )
    assert os.path.isfile(f), f"No GPM feature file at {f}"
    df = pl.read_csv(f)

    df = df.rename({"swath_edge_px": "cross_track_edge_px"})
    df = df.filter(~pl.col("touches_domain_edge"))
    df = df.with_columns(((pl.col("centroid_lon") + 180) % 360 - 180).alias("centroid_lon"))

    # Apply filtering
    df = df.filter(
        (pl.col("max_precip_mm_hr") >= maxpr_min)
        & (pl.col("max_precip_mm_hr") < maxpr_max)
        & (pl.col("size_px") >= size_px_min)
        & (pl.col("cross_track_edge_px") / pl.col("perimeter_px") <= edge_frac_max)
        & (pl.col("centroid_lat").abs() <= abs_lat_max)
    )
    return df