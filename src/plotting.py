import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
import seaborn as sns
from cycler import cycler
import scipy.stats as stats
import warnings

# Change the color palette
palette = sns.color_palette("Set2")
mpl.rcParams["axes.prop_cycle"] = cycler(color=palette)

# Ignore warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")

# Use monospace fonts everywhere
mpl.rcParams.update(
    {
        "font.family": "monospace",
        "font.monospace": ["DejaVu Sans Mono", "Courier New", "Liberation Mono", "monospace"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
         # Default font sizes
        "axes.labelsize": 12,   # xlabel / ylabel
        "xtick.labelsize": 11,  # x tick labels
        "ytick.labelsize": 11,  # y tick labels 
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.fontsize": 12,
        "legend.handlelength": 1.5
    }
)

def source_line_params(source):
    if source in ('GPM', 'IMERG'):
        color='black'
        lw = 2.5
        linestyle='solid'
    else:
        lw = 1.5
        linestyle='solid'
        match source:
            case 'GEOS' | 'GEOS-3km':
                color='#4C72B0'  # blue
            case 'gSAM' | 'gSAM-4km':
                color='#DD8452'  # orange
            case 'ICON' | 'ICON-SAP-5km':
                color='#55A868'  # green
            case 'MPAS' | 'MPAS-3km':
                color='#C44E52'  # red
            case 'SCREAM' | 'SCREAM-3km':
                color='#8172B3'  # purple
            case 'SHiELD' | 'SHiELD-3km':
                color='#937860'  # brown
            case _:
                color=None
    return {'color': color, 'lw': lw, 'linestyle': linestyle}

def array_midpoints(x):
    return (x[1:] + x[:-1]) / 2

def discrete_cmap(cmap_name="viridis", N=5, under_color=None):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, N))
    ans = ListedColormap(colors, name=f"{cmap_name}_{N}")
    if under_color is not None:
        ans.set_under(under_color)
    return ans


def combine_cmaps(cmap1, cmap2, split=0.5, N=256):
    n1 = int(N * split)
    n2 = N - n1
    colors1 = plt.get_cmap(cmap1)(np.linspace(0, 1, n1))
    colors2 = plt.get_cmap(cmap2)(np.linspace(0, 1, n2))
    return LinearSegmentedColormap.from_list(
        "combined", np.vstack((colors1, colors2))
    )

def plot_mean_stat(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    data_to_bin: np.ndarray,
    statistic: str,
    cmap: str,
    norm: plt.Normalize,
    density: bool = False,
    ax: plt.Axes = None,
    return_data: bool = False,
    figsize: tuple = (4,3),
    return_colorbar_separate: bool = False,
    cbar_kwargs: dict = None,
    **kwargs,
    ):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    data = stats.binned_statistic_2d(
        y_data,
        x_data,
        data_to_bin,
        statistic=statistic,
        bins=[y_bins, x_bins],
    ).statistic
    if density:
        data = 100 * data / np.nansum(data)
    x_mesh, y_mesh = np.meshgrid(x_bins, y_bins)
    im = ax.pcolormesh(
        x_mesh,
        y_mesh,
        data,
        cmap=cmap,
        norm=norm,
        **kwargs,
    )
        # Defaults that the user can override via cbar_kwargs.
    cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)

    if return_colorbar_separate:
        # Pull figsize out of cbar_kwargs if provided, else default based on orientation.
        orientation = cbar_kwargs.get('orientation', 'horizontal')
        default_cbar_figsize = (4, 0.4) if orientation == 'horizontal' else (0.4, 4)
        cbar_figsize = cbar_kwargs.pop('cbar_figsize', default_cbar_figsize)

        cbar_defaults = {'orientation': 'horizontal', 'extend': 'both'}
        cbar_defaults.update(cbar_kwargs)

        cbar_fig, cbar_ax = plt.subplots(figsize=cbar_figsize)
        cbar_fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            **cbar_defaults,
        )
        if cbar_defaults['orientation'] == 'horizontal':
            cbar_fig.subplots_adjust(left=0.05, right=0.95, bottom=0.5, top=0.9)
        else:
            cbar_fig.subplots_adjust(left=0.1, right=0.5, bottom=0.05, top=0.95)
    else:
        cbar_defaults = {'extend': 'both'}
        cbar_defaults.update(cbar_kwargs)
        cb = fig.colorbar(im, **cbar_defaults)

    if return_data and return_colorbar_separate:
        return fig, ax, cbar_fig, cbar_ax, data
    if return_data:
        return fig, ax, data
    else:
        return fig, ax

def gpm_area_factor():
    factor = (6_371 * np.deg2rad(0.05))**2
    return factor


def gpm_area_factor_accurate(latitude, res=0.05):
    """Per-pixel area [km2] of a res x res deg grid cell centred on `latitude`.

    gpm_area_factor() treats every pixel as a square of side R*dphi, which is only
    right at the equator: the meridional extent is indeed R*dphi everywhere, but the
    zonal extent shrinks as cos(lat). This returns the exact area of the spherical
    quadrilateral, R^2 * dlambda * (sin(lat_north) - sin(lat_south)), which is ~0.5%
    below the square value at 5 deg and ~6% below it at 20 deg, the edge of the
    tropical domain.

    Args:
        latitude: pixel or feature centre latitude in degrees; scalar, numpy array or
            polars Series (the result follows the input type).
        res: grid spacing in degrees, assumed equal in latitude and longitude.

    Returns:
        Per-pixel area in km2, of the same shape as `latitude`.
    """
    half = np.deg2rad(res) / 2
    return (
        6_371**2
        * np.deg2rad(res)
        * (np.sin(np.deg2rad(latitude) + half) - np.sin(np.deg2rad(latitude) - half))
    )
