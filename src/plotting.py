import numpy as np
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
    }
)


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
    **kwargs,
    ):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,3))
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
    cb = fig.colorbar(im, extend='both')
    if return_data:
        return fig, ax, data
    else:
        return fig, ax

def gpm_area_factor():
    factor = (6_371 * np.deg2rad(0.05))**2
    return factor
