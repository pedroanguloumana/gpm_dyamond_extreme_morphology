# Functions for loading GPM and DYAMOND data
import polars as pl
from src.gpm_regions import *
from src.paths import *
import os
import pickle
from src.paths import DATA_DIR

def legacy_gpm_feature_in_region_expr(region: GPMRegion) -> pl.Expr:
    lat_ok = (
        (pl.col("mean_latitude") >= region.lat_south)
        & (pl.col("mean_latitude") <= region.lat_north)
    )

    if region.lon_west <= region.lon_east:
        lon_ok = (
            (pl.col("mean_longitude") >= region.lon_west)
            & (pl.col("mean_longitude") <= region.lon_east)
        )
    else:
        lon_ok = (
            (pl.col("mean_longitude") >= region.lon_west)
            | (pl.col("mean_longitude") <= region.lon_east)
        )

    file_ok = pl.col("gpm_filename").str.contains(f"/{region.key}/", literal=True)

    return lat_ok & lon_ok & file_ok

def gpm_feature_in_region_expr(region: GPMRegion) -> pl.Expr:
    lat_ok = (
        (pl.col("centroid_lat") >= region.lat_south)
        & (pl.col("centroid_lat") <= region.lat_north)
    )

    if region.lon_west <= region.lon_east:
        lon_ok = (
            (pl.col("centroid_lon") >= region.lon_west)
            & (pl.col("centroid_lon") <= region.lon_east)
        )
    else:
        lon_ok = (
            (pl.col("centroid_lon") >= region.lon_west)
            | (pl.col("centroid_lon") <= region.lon_east)
        )

    file_ok = pl.col("source_file").str.contains(f"/{region.key}/", literal=True)

    return lat_ok & lon_ok & file_ok

def load_gpm_feature_stats(
    min_size: int = 5,
    maxpr_min: float = 10,
    # only_complete: bool = True,
    # months = None,
    ) -> pl.DataFrame:
    f = DATA_DIR / "gpm_features" / "merged.gpm.febs.csv"
    assert(os.path.isfile(f))
    df = pl.read_csv(f)

    trimmed_dfs = []

    for region_key in get_region_keys():
        region = get_region(region_key)
        trimmed_dfs.append(df.filter(gpm_feature_in_region_expr(region)))

    df = pl.concat(trimmed_dfs)

    # if only_complete:
    #     df = df.filter(pl.col("is_complete"))

    # if months is not None:
    #     df = df.filter(
    #         pl.col("observation_time")
    #         .str.slice(4, 2) # get the month part of the timestamp
    #         .cast(pl.Int64)
    #         .is_in(months)
    #     )

    df = df.filter(
        (pl.col("size_px") >= min_size)
        & (pl.col("max_precip_mm_hr") >= maxpr_min)
    )

    return df

def load_imerg_feature_stats(
    min_size: int = 5,
    maxpr_min: float = 10,
    # only_complete: bool = True,
    # months = None,
    ) -> pl.DataFrame:
    f = DATA_DIR / "gpm_features" / "merged.imerg.febs.csv"
    assert(os.path.isfile(f))
    df = pl.read_csv(f)

    # Unlike GPM, IMERG is stored as global monthly files, so there are no
    # overlapping regions to stitch together / de-duplicate.

    df = df.filter(
        (pl.col("size_px") >= min_size)
        & (pl.col("max_precip_mm_hr") >= maxpr_min)
    )

    return df

def legacy_load_gpm_feature_stats(
    # only_complete: bool = True,
    # min_size: int = 5,
    # maxpr_min: float = 10,
    months = None,
    ) -> pl.DataFrame:
    f =  DATA_DIR /  "old_data" / "merged.gpm_features.csv"
    assert(os.path.isfile(f))
    df = pl.read_csv(f)

    trimmed_dfs = []

    for region_key in get_region_keys():
        region = get_region(region_key)
        trimmed_dfs.append(df.filter(legacy_gpm_feature_in_region_expr(region)))

    df = pl.concat(trimmed_dfs)

    # if only_complete:
    #     df = df.filter(pl.col("is_complete"))
    
    if months is not None:
        df = df.filter(
            pl.col("observation_time")
            .str.slice(4, 2) # get the month part of the timestamp
            .cast(pl.Int64)
            .is_in(months)
        )

    # df = df.filter(
    #     (pl.col("number_pixels") >= min_size)
    #     & (pl.col("max_precip") >= maxpr_min)
    #     & (abs(pl.col("mean_latitude")) <= 20)
    # )
    # df = df.rename({"largest_10mmhr_cluster_size": "largest_10mmhr_cluster"})

    return df

def load_dyamond_feature_stats(model:str, only_complete: bool = True, maxpr_min: float = 10) -> pl.DataFrame:
    match model:
        case "GEOS-3km":
            f = DATA_DIR /  "merged.GEOS-3km_0p05deg_conservative_features.csv"
        case "gSAM-4km":
            f = DATA_DIR /  "merged.gSAM-4km_0p05deg_conservative_features.csv"
        case "ICON-SAP-5km":
            f = DATA_DIR /  "merged.ICON-SAP-5km_0p05deg_conservative_features.csv"
        case "SCREAM-3km":
            f = DATA_DIR /  "merged.SCREAM-3km_0p05deg_conservative_features.csv"
        case _:
            raise ValueError(f'Unsupported model {model} ')
    if not os.path.isfile(f):
        raise FileNotFoundError(f"No such file: {f}")
    df = pl.read_csv(f)
    if only_complete:
        df = df.filter(pl.col("is_complete"))
    df = df.filter(pl.col("max_precip") >= maxpr_min)
    
    return df

def load_combined_features(filename: str = "combined_features.pkl") -> dict[str, pl.DataFrame]:
    filename = DATA_DIR / filename
    if os.path.isfile(filename):
        print(f"Loading combined features from {filename}...")
        with open(filename, "rb") as f:
            return pickle.load(f)

    print(f"Combined features file {filename} not found. Loading individual datasets and combining...")
    combined: dict[str, pl.DataFrame] = {}

    combined["GPM"] = load_gpm_feature_stats(months=[1,2,3])

    for model in ["gSAM-4km", "ICON-SAP-5km", "SCREAM-3km", "GEOS-3km"]:
        combined[model.split("-")[0]] = load_dyamond_feature_stats(model)

    with open(filename, "wb") as f:
        pickle.dump(combined, f)

    return combined