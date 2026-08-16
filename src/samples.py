import numpy as np
from src.reading import read_gpm_feature_stats, read_dyamond_feature_stats

def load_gpm_dyamond_features():
    _MAXPR_MIN = 10.0
    _MAXPR_MAX = np.inf
    _RES = "0p05"
    _THRESH = "1p0"
    _EDGE_FRAC_MAX = 0.05
    _ABS_LAT_MAX = 20.0
    _SIZE_PX_MIN = 5
    _ACCUM = "15minaccum"

    feature_dict = {
        'GPM': read_gpm_feature_stats(threshold=_THRESH, res=_RES, maxpr_min=_MAXPR_MIN, maxpr_max=_MAXPR_MAX, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
        # 'ARPEGE': read_dyamond_feature_stats("ARPEGE-NH-2km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
        'GEOS': read_dyamond_feature_stats("GEOS-3km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
        # 'GRIST': read_dyamond_feature_stats("GRIST-5km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
        'gSAM': read_dyamond_feature_stats("gSAM-4km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
        'ICON': read_dyamond_feature_stats("ICON-SAP-5km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
        # 'MPAS': read_dyamond_feature_stats("MPAS-3km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
        'SCREAM': read_dyamond_feature_stats("SCREAM-3km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
        # 'SHiELD': read_dyamond_feature_stats("SHiELD-3km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),    
    }

    return feature_dict


def load_gpm_dyamond_edge_frac_max_sweep(
    edge_frac_maxes=(0.0, 0.05, 0.25, 0.5, 0.75, 1.0),
):
    cuts = dict(maxpr_min=10.0, maxpr_max=np.inf, abs_lat_max=20.0, size_px_min=5)
    res, thresh, accum = "0p05", "1p0", "15minaccum"

    out = {}
    for efm in edge_frac_maxes:
        d = {"GPM": read_gpm_feature_stats(threshold=thresh, res=res,
                                           edge_frac_max=efm, **cuts)}
        for name, model in DYAMOND_MODELS.items():
            d[name] = read_dyamond_feature_stats(model, accum=accum, res=res,
                                                 edge_frac_max=efm, **cuts)
        out[efm] = d
    return out


def load_gpm_dyamond_edge_frac_max_sweep():
    _MAXPR_MIN = 10.0
    _MAXPR_MAX = np.inf
    _RES = "0p05"
    _THRESH = "1p0"
    _ABS_LAT_MAX = 20.0
    _SIZE_PX_MIN = 5
    _ACCUM = "15minaccum"

    feature_dict = {}

    for _EDGE_FRAC_MAX in [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
        feature_dict[_EDGE_FRAC_MAX] = (
            {
                'GPM': read_gpm_feature_stats(threshold=_THRESH, res=_RES, maxpr_min=_MAXPR_MIN, maxpr_max=_MAXPR_MAX, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
                # 'ARPEGE': read_dyamond_feature_stats("ARPEGE-NH-2km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
                'GEOS': read_dyamond_feature_stats("GEOS-3km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
                # 'GRIST': read_dyamond_feature_stats("GRIST-5km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
                'gSAM': read_dyamond_feature_stats("gSAM-4km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
                'ICON': read_dyamond_feature_stats("ICON-SAP-5km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
                # 'MPAS': read_dyamond_feature_stats("MPAS-3km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
                'SCREAM': read_dyamond_feature_stats("SCREAM-3km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),
                # 'SHiELD': read_dyamond_feature_stats("SHiELD-3km", accum=_ACCUM, res=_RES, maxpr_max=_MAXPR_MAX, maxpr_min=_MAXPR_MIN, edge_frac_max=_EDGE_FRAC_MAX, abs_lat_max=_ABS_LAT_MAX, size_px_min=_SIZE_PX_MIN),    
            }
        )

    return feature_dict
