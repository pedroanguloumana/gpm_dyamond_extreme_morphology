# Functions for loading GPM and DYAMOND data
import polars as pl
import xarray as xr
from src.gpm_regions import *
from src.paths import *
import os
import pickle
from src.paths import DATA_DIR

# The DYAMOND runs are only analysed over +/-20 deg, so observations are cut to the same
# band by default to keep model/observation comparisons on a common domain.
TROPICS_ABS_LAT_MAX = 20.0

# The GPM 2Ku retrieval saturates at 300 mm/hr, so features pinned there are censored
# rather than measured: a few hundred of them are enough to move a decile contrast of
# max precip by ~10 percentage points. Pass maxpr_max=SATURATED_MAXPR_MIN to any loader
# to drop them (and the same cut to the models, which have no such cap).
SATURATED_MAXPR_MIN = 299.0

# region: GPM LOADING
GPM_DATA_DIR = DATA_DIR / "gpm_data"

# Some region boxes are not sampled by granules carrying their own key: H02p5 is a
# bookkeeping gap between the H02 and EUR boxes, and the granules covering it are H02.
_BOX_SAMPLED_BY: dict[str, str] = {"H02p5": "H02"}

# Region key as it appears at the end of a granule name, e.g.
# ".../GPM2Ku7_uw4_20150201.002002_to_20150201.002243_005264_AFC.nc"
_REGION_FROM_SOURCE_FILE = r"_([A-Za-z0-9]+)\.nc$"

def _fmt_threshold(threshold) -> str:
    """1.0 -> '1p0', 2.5 -> '2p5' (strings are passed through with '.' -> 'p')."""
    if isinstance(threshold, str):
        return threshold.replace(".", "p")
    return f"{float(threshold):.1f}".replace(".", "p")

def _fmt_res(res) -> str:
    """0.1 -> '0p1', 0.05 -> '0p05' (strings are passed through with '.' -> 'p')."""
    if isinstance(res, str):
        return res.replace(".", "p")
    return f"{float(res):g}".replace(".", "p")

def _fmt_accum(accum) -> str:
    """30 -> '30min', '15min' -> '15min' (accumulation window in minutes)."""
    if isinstance(accum, str):
        return accum if accum.endswith("min") else f"{accum}min"
    return f"{int(accum)}min"

def _fmt_accum_token(accum) -> str:
    """Accumulation token as it appears in a file name: 30 -> '30minaccum'.

    Instantaneous (non-accumulated) snapshots are named "instant", with no "accum".
    """
    if isinstance(accum, str) and accum.strip().lower() == "instant":
        return "instant"
    return f"{_fmt_accum(accum)}accum"

def load_gpm_feature_stats(
    threshold,
    res,
    maxpr_min: float = 0.0,
    maxpr_max=None,
    edge_frac_max: float = 1.0,# by default, accept everyone.
    abs_lat_max: float = TROPICS_ABS_LAT_MAX,
    area_min=0.0,
    size_px_min=5,
):
    """Load merged GPM feature stats for a given rain threshold and grid resolution.

    Each granule is labelled with the GPM region it was cut from (last token of
    ``source_file``), and neighbouring regions overlap because granules are cut with the
    wider "original" bounds. To avoid counting a feature once per overlapping region, a
    feature is kept only when its centroid falls inside the *non-overlapping* box of its
    own region (``src.gpm_regions``). Boxes are half-open in the north and east, so a
    feature sitting exactly on a shared edge is claimed by exactly one region.

    Args:
        threshold: precip threshold used to define features, e.g. 1.0 or "1p0".
        res: grid resolution in degrees, e.g. 0.1 or "0p1".
        maxpr_min: keep only features with max_precip_mm_hr >= this value.
        maxpr_max: keep only features with max_precip_mm_hr < this value. Set to
            ``SATURATED_MAXPR_MIN`` (299) to drop the features pinned at the 300 mm/hr
            retrieval cap, whose max precip is censored rather than measured. None
            keeps all.
        edge_frac_max: keep only features whose perimeter is less than this fraction
            swath edge, i.e. cross_track_edge_px / perimeter_px < edge_frac_max, so
            features substantially cut off by the swath edge are dropped. Set to None
            to keep them.
        abs_lat_max: keep only features with |centroid_lat| <= this value. The GPM
            granules reach |lat| ~ 60, well beyond the +/-20 deg DYAMOND domain, so
            this defaults to the tropical band the models are run over. None keeps all.
        area_min: keep only features with area_km2 >= this value. A pixel count is not
            comparable across grids (4 px is 124 km2 at 0.05 deg but 494 km2 at 0.1 deg),
            so use this to put every source on one physical size floor. None keeps all.
        size_px_min: keep only features with size_px >= this value. Unlike area_min this
            is a floor on the number of pixels a feature is resolved by, so it means a
            different physical size at each resolution. None keeps all.

    Returns:
        polars.DataFrame with an added "region" column.
    """
    f = GPM_DATA_DIR / f"merged_thresh{_fmt_threshold(threshold)}_res{_fmt_res(res)}.csv"
    assert os.path.isfile(f), f"No GPM feature file at {f}"

    df = pl.read_csv(f)
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

    df = df.filter(
        in_own_box,
        pl.col("max_precip_mm_hr") >= maxpr_min,
    )

    if maxpr_max is not None:
        df = df.filter(pl.col("max_precip_mm_hr") < maxpr_max)

    if area_min is not None:
        df = df.filter(pl.col("area_km2") >= area_min)

    if size_px_min is not None:
        df = df.filter(pl.col("size_px") >= size_px_min)

    if edge_frac_max is not None:
        df = df.filter(
            pl.col("cross_track_edge_px") / pl.col("perimeter_px") <= edge_frac_max
        )

    if abs_lat_max is not None:
        df = df.filter(pl.col("centroid_lat").abs() <= abs_lat_max)
    return df

# endregion

# region: GRIDDED (IMERG / DYAMOND) LOADING
IMERG_DATA_DIR = DATA_DIR / "imerg_data"
DYAMOND_DATA_DIR = DATA_DIR / "dyamond_data"

# Per-model subdirectory holding the same model at several accumulation windows.
DYAMOND_ACCUM_TEST_SUBDIR = "accumulation_window_test"

# Superseded DYAMOND feature stats, kept for comparison with the current pipeline.
ARCHIVED_DYAMOND_DATA_DIR = (
    DATA_DIR / "old_data" / "dyamond_features" / "archived_noinswathcounting"
)

# Pickled {source: DataFrame} from the first generation of the analysis (~7 GB).
ARCHIVED_COMBINED_FEATURES_FILE = DATA_DIR / "old_data" / "combined_features.pkl"

# IMERG is only processed at one accumulation window, so it is not an argument.
IMERG_ACCUM = "halfhour"


def _load_gridded_feature_stats(
    f,
    maxpr_min: float,
    edge_frac_max,
    drop_domain_edge: bool,
    to_lon180: bool,
    area_min=None,
    size_px_min=None,
    maxpr_max=None,
) -> pl.DataFrame:
    """Read one gridded (IMERG or DYAMOND) feature file and apply the shared cuts.

    Unlike GPM, these datasets are stored as global gridded files, so there are no
    overlapping regions to stitch together. Their swath columns come from sampling the
    global field onto synthetic DPR swaths, and the quantity GPM calls
    ``cross_track_edge_px`` is written as ``swath_edge_px`` in these files. It is
    renamed on read so every source shares the GPM name; the edge fraction is otherwise
    defined identically.
    """
    assert os.path.isfile(f), f"No feature file at {f}"

    df = pl.read_csv(f)
    df = df.rename({"swath_edge_px": "cross_track_edge_px"})
    df = df.filter(pl.col("max_precip_mm_hr") >= maxpr_min)

    if maxpr_max is not None:
        df = df.filter(pl.col("max_precip_mm_hr") < maxpr_max)

    if area_min is not None:
        df = df.filter(pl.col("area_km2") >= area_min)

    if size_px_min is not None:
        df = df.filter(pl.col("size_px") >= size_px_min)

    if edge_frac_max is not None:
        df = df.filter(
            pl.col("cross_track_edge_px") / pl.col("perimeter_px") <= edge_frac_max
        )

    if drop_domain_edge:
        df = df.filter(~pl.col("touches_domain_edge"))

    if to_lon180:
        df = df.with_columns(
            ((pl.col("centroid_lon") + 180) % 360 - 180).alias("centroid_lon")
        )
    return df


def load_imerg_feature_stats(
    threshold,
    maxpr_min: float = 0.0,
    edge_frac_max: float = 1.0,
    drop_domain_edge: bool = True,
    to_lon180: bool = True,
    abs_lat_max: float = TROPICS_ABS_LAT_MAX,
    area_min=None,
    maxpr_max=None,
):
    """Load IMERG feature stats for a given rain threshold.

    IMERG is produced on a single global 0.1 deg grid at half-hourly accumulation, so
    neither a resolution nor an accumulation argument is needed.

    Args:
        threshold: precip threshold used to define features, e.g. 1.0 or "1p0".
        maxpr_min: keep only features with max_precip_mm_hr >= this value.
        edge_frac_max: keep only features with cross_track_edge_px / perimeter_px <=
            this value, the same cut GPM gets. None keeps all.
        drop_domain_edge: drop features touching the domain edge, i.e. cut off by the
            +/-20 deg latitude limits or split at the lon 0/360 seam (features are not
            wrapped across it).
        to_lon180: rewrap centroid_lon from [0, 360) to [-180, 180) to match GPM.
        abs_lat_max: keep only features with |centroid_lat| <= this value, matching the
            +/-20 deg DYAMOND domain. The IMERG files are already cut to that band, so
            this is a no-op unless tightened. None keeps all.
        area_min: keep only features with area_km2 >= this value, the resolution
            independent size floor. Note the IMERG files already start at 4 px = 465 km2,
            so a smaller value does not add features back. None keeps all.
        maxpr_max: keep only features with max_precip_mm_hr < this value, the analogue
            of the GPM saturation cut. None keeps all.

    Returns:
        polars.DataFrame
    """
    f = IMERG_DATA_DIR / (
        f"imerg_features_{IMERG_ACCUM}_thresh{_fmt_threshold(threshold)}.csv"
    )
    df = _load_gridded_feature_stats(
        f, maxpr_min, edge_frac_max, drop_domain_edge, to_lon180, area_min,
        maxpr_max=maxpr_max,
    )
    if abs_lat_max is not None:
        df = df.filter(pl.col("centroid_lat").abs() <= abs_lat_max)
    return df


def load_dyamond_feature_stats(
    model: str,
    accum="15min",
    res="0p05",
    maxpr_min: float = 0.0,
    edge_frac_max: float = 1.0,
    drop_domain_edge: bool = True,
    to_lon180: bool = True,
    area_min=None,
    size_px_min=5,
    maxpr_max=None,
):
    """Load DYAMOND feature stats for one model, accumulation window and resolution.

    Files live at ``data/dyamond_data/{model}/{model}_features_{accum}_{res}deg.csv``,
    e.g. "SCREAM-3km/SCREAM-3km_features_30minaccum_0p1deg.csv".

    Args:
        model: DYAMOND model name as it appears in the directory, e.g. "SCREAM-3km".
        accum: accumulation window, e.g. "30min", "15min" or 30.
        res: grid resolution in degrees, e.g. "0p1", "0p05" or 0.05.
        maxpr_min: keep only features with max_precip_mm_hr >= this value.
        edge_frac_max: keep only features with cross_track_edge_px / perimeter_px <=
            this value, the same cut GPM gets. None keeps all.
        drop_domain_edge: drop features touching the domain edge, i.e. cut off by the
            +/-20 deg latitude limits or split at the lon 0/360 seam (features are not
            wrapped across it).
        to_lon180: rewrap centroid_lon from [0, 360) to [-180, 180) to match GPM.
        area_min: keep only features with area_km2 >= this value, a floor that means the
            same physical size at every resolution unlike a pixel count. None keeps all.
        size_px_min: keep only features with size_px >= this value, a floor on how many
            pixels a feature is resolved by rather than on its physical size, so it does
            not mean the same thing at every resolution. None keeps all.
        maxpr_max: keep only features with max_precip_mm_hr < this value. The models
            have no retrieval cap, so this exists to apply the same cut GPM needs
            (``SATURATED_MAXPR_MIN``) to both sides of a comparison. None keeps all.

    Returns:
        polars.DataFrame
    """
    f = (
        DYAMOND_DATA_DIR
        / model
        / f"{model}_features_{_fmt_accum_token(accum)}_{_fmt_res(res)}deg.csv"
    )
    return _load_gridded_feature_stats(
        f, maxpr_min, edge_frac_max, drop_domain_edge, to_lon180, area_min, size_px_min,
        maxpr_max,
    )


def load_dyamond_accum_test_feature_stats(
    model: str = "gSAM-4km",
    accum="30min",
    res="0p1",
    maxpr_min: float = 10.0,
    edge_frac_max: float = 0.05,
    drop_domain_edge: bool = True,
    to_lon180: bool = True,
    area_min=None,
    size_px_min=5,
):
    """Load feature stats from a model's accumulation-window test set.

    Same files and cuts as :func:`load_dyamond_feature_stats`, but from the
    ``accumulation_window_test`` subdirectory, where one model is processed at several
    accumulation windows so the effect of the window can be isolated. Only gSAM-4km has
    this set so far, at "instant", "15min" and "30min" on the 0.1 deg grid.

    Args:
        model: DYAMOND model name as it appears in the directory, e.g. "gSAM-4km".
        accum: accumulation window, e.g. "instant" for the un-accumulated snapshots,
            or "15min", "30min", 30.
        res: grid resolution in degrees, e.g. "0p1", "0p05" or 0.05.
        maxpr_min: keep only features with max_precip_mm_hr >= this value.
        edge_frac_max: keep only features with cross_track_edge_px / perimeter_px <=
            this value, the same cut GPM gets. None keeps all.
        drop_domain_edge: drop features touching the domain edge, i.e. cut off by the
            +/-20 deg latitude limits or split at the lon 0/360 seam (features are not
            wrapped across it).
        to_lon180: rewrap centroid_lon from [0, 360) to [-180, 180) to match GPM.
        area_min: keep only features with area_km2 >= this value. None keeps all.
        size_px_min: keep only features with size_px >= this value, a pixel-count floor
            rather than a physical-size one. None keeps all.

    Returns:
        polars.DataFrame
    """
    f = (
        DYAMOND_DATA_DIR
        / model
        / DYAMOND_ACCUM_TEST_SUBDIR
        / f"{model}_features_{_fmt_accum_token(accum)}_{_fmt_res(res)}deg.csv"
    )
    return _load_gridded_feature_stats(
        f, maxpr_min, edge_frac_max, drop_domain_edge, to_lon180, area_min, size_px_min
    )


def load_archived_dyamond_feature_stats(
    model: str,
    res="0p1",
    maxpr_min: float = 10.0,
    swath_edge_px_max=None,
    drop_domain_edge: bool = True,
    to_lon180: bool = True,
    area_min=None,
    size_px_min=5,
):
    """Load archived (pre in-swath-counting) DYAMOND feature stats for one model.

    Files live at ``data/old_data/dyamond_features/archived_noinswathcounting/
    {model}_features_{res}deg.csv``. These predate the current pipeline: features were
    not counted per swath, so there is no ``px_in_swath`` column, and crucially no
    ``perimeter_px`` either, which is why the ``edge_frac_max`` cut of the other
    gridded loaders is replaced here by an absolute ``swath_edge_px_max``. All models
    are available at 0.1 deg; only GEOS-3km, ICON-SAP-5km and SHiELD-3km at 0.05 deg.

    Args:
        model: DYAMOND model name as it appears in the file, e.g. "SCREAM-3km".
        res: grid resolution in degrees, e.g. "0p1", "0p05" or 0.05.
        maxpr_min: keep only features with max_precip_mm_hr >= this value.
        swath_edge_px_max: keep only features with cross_track_edge_px <= this value; 0
            drops every feature touching a swath edge, the closest analogue available
            here to ``edge_frac_max=0``. None keeps all.
        drop_domain_edge: drop features touching the domain edge, i.e. cut off by the
            +/-20 deg latitude limits or split at the lon 0/360 seam (features are not
            wrapped across it).
        to_lon180: rewrap centroid_lon from [0, 360) to [-180, 180) to match GPM.
        area_min: keep only features with area_km2 >= this value. These files were built
            with a 4 px minimum, which is 465 km2 at 0.1 deg but only 116 km2 at 0.05 deg,
            so a floor below 465 km2 cannot be honoured at 0.1 deg. None keeps all.
        size_px_min: keep only features with size_px >= this value. These files were
            built with a 4 px minimum, so a floor below that adds no features back.
            None keeps all.

    Returns:
        polars.DataFrame
    """
    f = ARCHIVED_DYAMOND_DATA_DIR / f"{model}_features_{_fmt_res(res)}deg.csv"
    df = _load_gridded_feature_stats(
        f, maxpr_min, None, drop_domain_edge, to_lon180, area_min, size_px_min
    )
    if swath_edge_px_max is not None:
        df = df.filter(pl.col("cross_track_edge_px") <= swath_edge_px_max)
    return df


def load_pickled_archived_features(keys=None) -> dict[str, pl.DataFrame]:
    """Load the pickled {source: DataFrame} dict from the first-generation analysis.

    Keys are "GPM", "GEOS", "gSAM", "ICON" and "SCREAM" (no SHiELD or MPAS), each a
    polars DataFrame. GPM carries ~50 columns the models do not (echo top heights,
    convective/stratiform splits, more precip thresholds), so the frames do not share
    a schema.

    These predate the CSV pipeline and share none of its schema either: there is no
    ``area_km2`` (only ``number_pixels``), precip is ``max_precip`` rather than
    ``max_precip_mm_hr``, features are located by ``mean_latitude`` /
    ``mean_longitude``, and completeness is a single ``is_complete`` flag instead of
    the swath- and domain-edge pixel counts. So none of the cuts the other loaders
    apply are applied here, and the frames are returned exactly as pickled.

    The file is ~7 GB on disk and unpickles to a comparable amount of memory, with
    3-6 M rows per model, so pass ``keys`` to keep only the sources you need.

    Args:
        keys: source names to keep, e.g. ["GPM", "GEOS"]. None keeps all. Note that
            the whole pickle is read regardless; this only limits what is retained.

    Returns:
        dict mapping source name to polars.DataFrame.
    """
    f = ARCHIVED_COMBINED_FEATURES_FILE
    assert os.path.isfile(f), f"No pickled feature file at {f}"

    with open(f, "rb") as fh:
        features = pickle.load(fh)

    if keys is not None:
        missing = [k for k in keys if k not in features]
        assert not missing, f"No such source(s) {missing} in {f}, have {list(features)}"
        features = {k: features[k] for k in keys}
    return features

# endregion 