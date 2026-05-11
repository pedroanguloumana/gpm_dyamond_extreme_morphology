import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class BinnedQuantileDelta:
    """Compute and plot binned quantile differences on a 2D grid."""

    @staticmethod
    def _pair_columns(col1, col2):
        """Pack two columns into an object array of tuples.

        Needed because scipy.binned_statistic_2d only accepts 1D values
        arrays. Packing pairs as tuples in an object array lets us pass
        both the sorting and diffing data through a single argument.
        """
        assert len(col1) == len(col2)
        arr = np.empty(len(col1), dtype=object)
        for i, pair in enumerate(zip(col1, col2)):
            arr[i] = pair
        return arr
    
    @staticmethod
    def _array_midpoints(arr):
        """Compute bin midpoints from bin edges."""
        return (arr[:-1] + arr[1:]) / 2

    def __init__(
        self,
        data_to_sort,
        data_to_diff,
        x_data,
        y_data,
        x_bins,
        y_bins,
        low_quantile=(0.0, 0.1),
        high_quantile=(0.9, 1.0),
        percent_diff: bool = False,
        samp_min: int = 10,
        n_perm: int = 100,
    ):
        self.data_array = self._pair_columns(data_to_sort, data_to_diff)
        self.x_data = x_data
        self.y_data = y_data
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.percent_diff = percent_diff
        self.samp_min = samp_min
        self.n_perm = n_perm

    def _quantile_indices(self, n):
        """Convert quantile bounds to array indices for a sample of size n."""
        low_start = round(n * self.low_quantile[0])
        low_end = round(n * self.low_quantile[1])
        high_start = round(n * self.high_quantile[0])
        high_end = round(n * self.high_quantile[1])
        return low_start, low_end, high_start, high_end

    def binned_quantile_delta(self, tuple_arr, percent_diff=None):
        if percent_diff is None:
            percent_diff = self.percent_diff

        arr = np.array(tuple_arr)
        if arr.shape[0] < self.samp_min:
            return np.nan

        sorted_arr = arr[arr[:, 0].argsort()]
        li0, li1, hi0, hi1 = self._quantile_indices(arr.shape[0])

        low = np.nanmean(sorted_arr[li0:li1, 1])
        high = np.nanmean(sorted_arr[hi0:hi1, 1])
        delta = high - low

        if percent_diff:
            delta = 100 * delta / low
        return delta

    def binned_quantile_delta_pval(self, tuple_arr):
        arr = np.array(tuple_arr)
        if arr.shape[0] < self.samp_min:
            return np.nan

        delta = self.binned_quantile_delta(tuple_arr, percent_diff=False)
        li0, li1, hi0, hi1 = self._quantile_indices(arr.shape[0])

        rng = np.random.default_rng(seed=67)
        random_deltas = np.empty(self.n_perm)
        for i in range(self.n_perm):
            perm = rng.permutation(arr)
            random_deltas[i] = np.nanmean(perm[hi0:hi1, 1]) - np.nanmean(perm[li0:li1, 1])

        p = (np.sum(np.abs(random_deltas) >= np.abs(delta)) + 1) / (self.n_perm + 1)
        return p

    def plot(self, xlabel, ylabel, title, cmap, norm, xscale='log', ax=None, add_text=True, return_p=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        # Compute delta statistic on 2D grid
        s = stats.binned_statistic_2d(
            self.y_data, self.x_data, self.data_array,
            statistic=self.binned_quantile_delta,
            bins=[self.y_bins, self.x_bins],
        ).statistic.astype(float)

        # Compute p-values
        p = stats.binned_statistic_2d(
            self.y_data, self.x_data, self.data_array,
            statistic=self.binned_quantile_delta_pval,
            bins=[self.y_bins, self.x_bins],
        ).statistic.astype(float)

        # Plot heatmap
        x_mesh, y_mesh = np.meshgrid(self.x_bins, self.y_bins)
        im = ax.pcolormesh(x_mesh, y_mesh, s, cmap=cmap, norm=norm)
        fig.colorbar(im)

        # Hatch significant bins
        xx, yy = np.meshgrid(self._array_midpoints(self.x_bins), self._array_midpoints(self.y_bins))
        ax.contourf(
            xx, yy, (p <= 0.05).astype(float),
            levels=[0.5, 1.5],
            colors='none',
            hatches=['...'],
            linewidths=0,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale(xscale)
        ax.grid()

        sig_mean = np.nanmean(s[p <= 0.05])
        if add_text:
            ax.text(
                0.95, 0.95,
                f'Avg:{sig_mean:.2f}%',
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=12,
                bbox=dict(facecolor='wheat', edgecolor='black', boxstyle='round,pad=0.3',)
            )
        print(f'{title} significant-bin mean: {sig_mean:.3f}')
        if return_p:
            return fig, ax, s, sig_mean, p
        return fig, ax, s, sig_mean