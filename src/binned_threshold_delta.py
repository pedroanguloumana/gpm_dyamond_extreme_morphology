"""Binned threshold-split contrast of a response between low and high groups.

Standalone: run ``python binned_threshold_delta.py`` (or ``pytest`` on this file) to
execute the tests at the bottom.
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def summarise_grid(s, p, counts, alpha=0.05):
    """Summary statistics of a computed (statistic, p-value, count) grid.

    ``count_weighted_sig_mean`` is the headline number to report: the plain
    ``sig_mean`` is an unweighted mean over only the bins that pass ``p <= alpha``,
    so it is selected on significance and drifts with sample size (a smaller sample
    keeps only the bins with the largest values, biasing it up). Weighting by how
    many features each bin holds is far more stable.
    """
    
    filled = ~np.isnan(s)
    sig = filled & (p <= alpha)
    keep = sig & ~np.isnan(counts)
    weighted = (
        float(np.ma.average(np.ma.masked_array(s, ~keep),
                            weights=np.ma.masked_array(counts, ~keep)))
        if keep.any() else np.nan
    )
    return {
        "filled_bins": int(filled.sum()),
        "sig_bins": int(sig.sum()),
        "count_weighted_sig_mean": weighted,
        "sig_mean": float(np.nanmean(s[sig])) if sig.any() else np.nan,
        "sig_median": float(np.nanmedian(s[sig])) if sig.any() else np.nan,
        "n_features": float(np.nansum(counts)),
    }


def _bin_groups(y_data, x_data, y_bins, x_bins):
    """Group row indices by 2D bin.

    Returns ``([(flat_bin, row_indices), ...], counts)``, where ``flat_bin`` indexes a
    flattened (ny, nx) grid and ``counts`` is the per-bin sample count in grid shape.
    Rows outside the bin edges are dropped. Bins are listed in ascending flat order and
    row indices keep their original order within a bin, so iteration is deterministic.
    """
    y_data = np.asarray(y_data, float)
    x_data = np.asarray(x_data, float)
    y_bins = np.asarray(y_bins, float)
    x_bins = np.asarray(x_bins, float)
    ny, nx = y_bins.size - 1, x_bins.size - 1

    binned = stats.binned_statistic_2d(
        y_data, x_data, None,
        statistic="count",
        bins=[y_bins, x_bins],
        expand_binnumbers=True,
    )
    iy, ix = binned.binnumber
    in_range = (iy >= 1) & (iy <= ny) & (ix >= 1) & (ix <= nx)

    flat = np.where(in_range, (iy - 1) * nx + (ix - 1), -1)
    order = np.flatnonzero(in_range)
    order = order[np.argsort(flat[order], kind="stable")]
    if order.size == 0:
        return [], binned.statistic.astype(float)

    b = flat[order]
    starts = np.flatnonzero(np.r_[True, np.diff(b) != 0])
    ends = np.r_[starts[1:], b.size]
    groups = [(int(b[s]), order[s:e]) for s, e in zip(starts, ends)]
    return groups, binned.statistic.astype(float)


class BinnedThresholdDelta:
    """Difference in mean ``data_to_diff`` between low- and high-``data_to_sort``
    groups, computed in 2D bins of ``(x_data, y_data)``.

    Groups are defined by value, not by rank: low is ``data_to_sort <= low_threshold``,
    high is ``data_to_sort >= high_threshold``. Ties therefore cannot straddle a group
    boundary and the result does not depend on row order -- which is the whole reason
    for using thresholds on a variable like rho = largest_core / n_conv, ~70% of whose
    values sit at exactly 1.0.

    Features strictly between the thresholds belong to neither group and take no part
    in the contrast or in its null distribution. Bins where either group has fewer than
    ``group_min`` members are left blank rather than filled with a contrast the data
    cannot support.

    The p-value comes from permuting the pooled low+high values and re-splitting at the
    observed group sizes, so the null is "data_to_diff carries no information about
    which side of the thresholds a feature falls on".
    """

    def __init__(
        self,
        data_to_sort,
        data_to_diff,
        x_data,
        y_data,
        x_bins,
        y_bins,
        low_threshold,
        high_threshold,
        percent_diff: bool = True,
        group_min: int = 20,
        n_perm: int = 200,
        seed: int = 67,
    ):
        """
        Args:
            data_to_sort: variable the low/high split is made on, e.g. rho.
            data_to_diff: variable differenced between the groups, e.g. max precip.
            x_data, y_data: coordinates each sample is binned by.
            x_bins, y_bins: bin edges. The returned grids are (y, x), i.e. rows are
                ``y_bins`` and columns are ``x_bins``, matching pcolormesh.
            low_threshold, high_threshold: group bounds; low is ``<= low_threshold``,
                high is ``>= high_threshold``.
            percent_diff: report the difference as a percentage of the low-group mean.
            group_min: minimum members in *each* group for a bin to be filled.
            n_perm: permutation draws for the p-value; 0 skips the test and leaves the
                p-grid as NaN.
            seed: permutation seed.
        """
        assert low_threshold < high_threshold, (
            "low_threshold must be < high_threshold so the two groups are disjoint"
        )
        assert group_min >= 1, "group_min must be at least 1"

        sort = np.asarray(data_to_sort, dtype=float)
        diff = np.asarray(data_to_diff, dtype=float)
        x = np.asarray(x_data, dtype=float)
        y = np.asarray(y_data, dtype=float)
        n = sort.size
        assert diff.size == n and x.size == n and y.size == n, (
            "data_to_sort, data_to_diff, x_data and y_data must be the same length"
        )

        # Drop unusable rows before binning, so counts are the number of features each
        # delta actually rests on. A NaN in data_to_sort would otherwise fall silently
        # into neither group while still being counted.
        good = np.isfinite(sort) & np.isfinite(diff) & np.isfinite(x) & np.isfinite(y)
        self.sort = sort[good]
        self.diff = diff[good]
        self.x_data = x[good]
        self.y_data = y[good]
        self.n_dropped = int((~good).sum())

        self.x_bins = np.asarray(x_bins, dtype=float)
        self.y_bins = np.asarray(y_bins, dtype=float)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.percent_diff = percent_diff
        self.group_min = group_min
        self.n_perm = n_perm
        self.seed = seed

        # Filled by compute(): members of the two groups per bin, as opposed to every
        # feature in the bin. Use it as the weight if the middle band is large.
        self.group_counts = None

    @staticmethod
    def _array_midpoints(arr):
        """Bin midpoints from bin edges."""
        return (arr[:-1] + arr[1:]) / 2

    def _pval(self, low, high, delta, rng):
        """Two-sided permutation p-value for one bin.

        The pool is the two groups only, not the whole bin: features between the
        thresholds take no part in the observed contrast, so they take none in its null
        either. The pool is sorted first so the p-value does not depend on the order the
        rows arrived in.
        """
        pool = np.sort(np.concatenate([low, high]))
        perm = rng.permuted(np.tile(pool, (self.n_perm, 1)), axis=1)
        null = perm[:, low.size:].mean(axis=1) - perm[:, : low.size].mean(axis=1)
        return float((np.sum(np.abs(null) >= abs(delta)) + 1) / (self.n_perm + 1))

    def compute(self):
        """Return ``(s, p, counts)`` on the (y_bins, x_bins) grid.

        ``s`` is the delta (percent if ``percent_diff``), ``p`` its permutation
        p-value, and ``counts`` the number of features in each bin -- *every* feature,
        including those between the thresholds. ``self.group_counts`` holds the number
        that fell in one of the two groups.
        """
        ny, nx = self.y_bins.size - 1, self.x_bins.size - 1
        groups, counts = _bin_groups(self.y_data, self.x_data, self.y_bins, self.x_bins)

        
        s = np.full(ny * nx, np.nan)
        p = np.full(ny * nx, np.nan)
        gn = np.zeros(ny * nx)
        rng = np.random.default_rng(self.seed)

        for b, idx in groups:
            sort, diff = self.sort[idx], self.diff[idx]
            low = diff[sort <= self.low_threshold]
            high = diff[sort >= self.high_threshold]
            gn[b] = low.size + high.size
            if low.size < self.group_min or high.size < self.group_min:
                continue

            delta = high.mean() - low.mean()
            if self.percent_diff:
                low_mean = low.mean()
                if low_mean == 0:
                    # A percentage of zero is not defined; blank rather than return inf.
                    continue
                value = 100 * delta / low_mean
            else:
                value = delta

            if self.n_perm:
                p[b] = self._pval(low, high, delta, rng)
            s[b] = value

        shape = (ny, nx)
        self.group_counts = gn.reshape(shape)
        return s.reshape(shape), p.reshape(shape), counts

    @staticmethod
    def summarise(s, p, counts, alpha=0.05):
        """Summary statistics of a computed grid. See ``summarise_grid``."""
        return summarise_grid(s, p, counts, alpha=alpha)

    def plot(self, xlabel, ylabel, title, cmap, norm, xscale="log", ax=None,
             add_text=False, return_p=False, return_counts=False, weight_by_count=True,
             alpha=0.05):
        """Plot the binned delta and report its summary.

        Significant bins are hatched; the headline is the count-weighted mean over them.

        Args:
            weight_by_count: report the count-weighted mean over significant bins as the
                headline number. Set False for the unweighted mean. Both are printed.
            return_p: also return the p-value grid.
            return_counts: also return the per-bin counts.

        Returns:
            ``(fig, ax, s, headline)``, plus ``p`` and ``counts`` if requested.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        s, p, counts = self.compute()
        summary = self.summarise(s, p, self.group_counts, alpha=alpha)
        summary["n_features"] = float(np.nansum(counts))   # every feature in the grid
        summary["n_in_contrast"] = float(np.nansum(self.group_counts))
        headline = (
            summary["count_weighted_sig_mean"] if weight_by_count else summary["sig_mean"]
        )

        x_mesh, y_mesh = np.meshgrid(self.x_bins, self.y_bins)
        im = ax.pcolormesh(x_mesh, y_mesh, s, cmap=cmap, norm=norm)
        fig.colorbar(im)

        xx, yy = np.meshgrid(self._array_midpoints(self.x_bins),
                             self._array_midpoints(self.y_bins))
        ax.contourf(
            xx, yy, (p <= alpha).astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["..."],
            linewidths=0,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale(xscale)
        ax.grid()

        unit = "%" if self.percent_diff else ""
        if add_text:
            label = "Wtd avg" if weight_by_count else "Avg"
            ax.text(
                0.95, 0.95,
                f"{label}:{headline:.2f}{unit}",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=12,
                bbox=dict(facecolor="wheat", edgecolor="black",
                          boxstyle="round,pad=0.3"),
            )
        print(
            f"{title} count-weighted significant-bin mean: "
            f"{summary['count_weighted_sig_mean']:.3f}{unit} "
            f"(unweighted: {summary['sig_mean']:.3f}{unit}, "
            f"{summary['sig_bins']}/{summary['filled_bins']} bins significant)"
        )

        out = [fig, ax, s, headline]
        if return_p:
            out.append(p)
        if return_counts:
            out.append(counts)
        return tuple(out)


# region: TESTS

def _one_bin(n_low=30, n_high=30, n_mid=0, low_val=10.0, high_val=20.0,
             low_rho=0.5, high_rho=1.0, mid_rho=0.9):
    """One bin's worth of samples, all landing at (x, y) = (5, 0.5)."""
    sort = np.r_[np.full(n_low, low_rho), np.full(n_high, high_rho),
                 np.full(n_mid, mid_rho)]
    diff = np.r_[np.full(n_low, low_val), np.full(n_high, high_val),
                 np.full(n_mid, 1e6)]  # absurd, to catch the middle band leaking in
    n = sort.size
    return dict(data_to_sort=sort, data_to_diff=diff,
                x_data=np.full(n, 5.0), y_data=np.full(n, 0.5),
                x_bins=np.array([0.0, 10.0]), y_bins=np.array([0.0, 1.0]),
                low_threshold=0.8, high_threshold=1.0)


def test_delta_absolute_and_percent():
    kw = _one_bin(low_val=10.0, high_val=20.0)
    s, _, _ = BinnedThresholdDelta(**kw, percent_diff=False, group_min=5,
                                   n_perm=0).compute()
    assert np.allclose(s, 10.0)

    s, _, _ = BinnedThresholdDelta(**kw, percent_diff=True, group_min=5,
                                   n_perm=0).compute()
    assert np.allclose(s, 100.0)


def test_middle_band_excluded_from_delta():
    """Features between the thresholds must not touch the contrast."""
    kw = _one_bin(n_mid=50)  # their diff value is 1e6
    s, _, _ = BinnedThresholdDelta(**kw, percent_diff=True, group_min=5,
                                   n_perm=0).compute()
    assert np.allclose(s, 100.0)


def test_threshold_boundaries_are_inclusive():
    """A value exactly on a threshold joins that group."""
    kw = _one_bin(low_rho=0.8, high_rho=1.0)  # both sit on their bounds
    s, _, _ = BinnedThresholdDelta(**kw, percent_diff=False, group_min=5,
                                   n_perm=0).compute()
    assert np.allclose(s, 10.0)


def test_group_min_blanks_the_bin():
    kw = _one_bin(n_low=4, n_high=30)
    s, p, counts = BinnedThresholdDelta(**kw, group_min=5, n_perm=0).compute()
    assert np.all(np.isnan(s)) and np.all(np.isnan(p))
    assert counts[0, 0] == 34  # the features are still counted


def test_group_counts_excludes_middle_band():
    kw = _one_bin(n_low=30, n_high=30, n_mid=40)
    btd = BinnedThresholdDelta(**kw, group_min=5, n_perm=0)
    _, _, counts = btd.compute()
    assert counts[0, 0] == 100
    assert btd.group_counts[0, 0] == 60


def test_nonfinite_rows_dropped():
    kw = _one_bin(n_low=30, n_high=30)
    kw["data_to_diff"] = kw["data_to_diff"].copy()
    kw["data_to_diff"][0] = np.nan
    kw["data_to_sort"] = kw["data_to_sort"].copy()
    kw["data_to_sort"][-1] = np.nan
    btd = BinnedThresholdDelta(**kw, group_min=5, n_perm=0)
    assert btd.n_dropped == 2
    _, _, counts = btd.compute()
    assert counts[0, 0] == 58


def test_grid_orientation_is_y_by_x():
    """Rows are y_bins, columns are x_bins -- the layout pcolormesh expects."""
    kw = _one_bin(n_low=10, n_high=10)
    n = kw["data_to_sort"].size
    kw["x_data"] = np.full(n, 25.0)   # third x bin
    kw["y_data"] = np.full(n, 0.15)   # second y bin
    kw["x_bins"] = np.array([0.0, 10.0, 20.0, 30.0])
    kw["y_bins"] = np.array([0.0, 0.1, 0.2])
    s, _, counts = BinnedThresholdDelta(**kw, percent_diff=False, group_min=5,
                                        n_perm=0).compute()
    assert s.shape == (2, 3) and counts.shape == (2, 3)
    assert np.isfinite(s[1, 2])
    assert np.isnan(np.delete(s, 5))  # every other bin is empty
    assert counts[1, 2] == n


def test_out_of_range_samples_ignored():
    kw = _one_bin(n_low=30, n_high=30)
    n = kw["data_to_sort"].size
    kw["x_data"] = kw["x_data"].copy()
    kw["x_data"][:5] = 999.0  # outside x_bins
    _, _, counts = BinnedThresholdDelta(**kw, group_min=5, n_perm=0).compute()
    assert counts.sum() == n - 5


def test_row_order_does_not_change_the_answer():
    """The point of a threshold split: no rank ties to resolve, so no order effect."""
    kw = _one_bin(n_low=40, n_high=40, n_mid=20)
    s0, p0, c0 = BinnedThresholdDelta(**kw, group_min=5, n_perm=50).compute()

    rng = np.random.default_rng(0)
    perm = rng.permutation(kw["data_to_sort"].size)
    shuffled = {k: (v[perm] if isinstance(v, np.ndarray) and v.size == perm.size else v)
                for k, v in kw.items()}
    s1, p1, c1 = BinnedThresholdDelta(**shuffled, group_min=5, n_perm=50).compute()

    assert np.array_equal(s0, s1, equal_nan=True)
    assert np.array_equal(p0, p1, equal_nan=True)
    assert np.array_equal(c0, c1)


def test_identical_groups_give_p_of_one():
    """delta = 0, so every permuted |delta| is >= it: p is exactly 1."""
    kw = _one_bin(low_val=10.0, high_val=10.0)
    s, p, _ = BinnedThresholdDelta(**kw, percent_diff=False, group_min=5,
                                   n_perm=200).compute()
    assert np.allclose(s, 0.0)
    assert np.allclose(p, 1.0)


def test_clean_separation_gives_minimum_p():
    """No overlap between the groups, so no permutation can reach the observed delta."""
    rng = np.random.default_rng(1)
    n = 60
    sort = np.r_[np.full(n, 0.5), np.full(n, 1.0)]
    diff = np.r_[rng.normal(10, 0.1, n), rng.normal(20, 0.1, n)]
    btd = BinnedThresholdDelta(
        data_to_sort=sort, data_to_diff=diff,
        x_data=np.full(2 * n, 5.0), y_data=np.full(2 * n, 0.5),
        x_bins=np.array([0.0, 10.0]), y_bins=np.array([0.0, 1.0]),
        low_threshold=0.8, high_threshold=1.0, group_min=5, n_perm=200,
    )
    _, p, _ = btd.compute()
    assert np.allclose(p, 1 / 201)


def test_pvalue_is_reproducible():
    kw = _one_bin(n_low=40, n_high=40)
    kw["data_to_diff"] = np.random.default_rng(2).normal(10, 3, 80)
    a = BinnedThresholdDelta(**kw, group_min=5, n_perm=100, seed=5).compute()[1]
    b = BinnedThresholdDelta(**kw, group_min=5, n_perm=100, seed=5).compute()[1]
    assert np.array_equal(a, b, equal_nan=True)


def test_n_perm_zero_skips_the_test():
    kw = _one_bin()
    s, p, _ = BinnedThresholdDelta(**kw, group_min=5, n_perm=0).compute()
    assert np.all(np.isfinite(s)) and np.all(np.isnan(p))


def test_thresholds_must_be_disjoint():
    kw = _one_bin()
    kw["low_threshold"], kw["high_threshold"] = 1.0, 0.8
    try:
        BinnedThresholdDelta(**kw)
    except AssertionError:
        return
    raise AssertionError("expected low_threshold >= high_threshold to be rejected")


def test_mismatched_lengths_rejected():
    kw = _one_bin()
    kw["data_to_diff"] = kw["data_to_diff"][:-1]
    try:
        BinnedThresholdDelta(**kw)
    except AssertionError:
        return
    raise AssertionError("expected mismatched input lengths to be rejected")


def test_summarise_reports_significant_bins():
    """Two bins, one with a real contrast and one with none."""
    n = 40
    sort = np.tile(np.r_[np.full(n, 0.5), np.full(n, 1.0)], 2)
    diff = np.r_[np.full(n, 10.0), np.full(n, 20.0),   # bin 1: contrast
                 np.full(n, 10.0), np.full(n, 10.0)]   # bin 2: none
    x = np.r_[np.full(2 * n, 5.0), np.full(2 * n, 15.0)]
    btd = BinnedThresholdDelta(
        data_to_sort=sort, data_to_diff=diff, x_data=x,
        y_data=np.full(4 * n, 0.5),
        x_bins=np.array([0.0, 10.0, 20.0]), y_bins=np.array([0.0, 1.0]),
        low_threshold=0.8, high_threshold=1.0, group_min=5, n_perm=200,
    )
    s, p, counts = btd.compute()
    summary = btd.summarise(s, p, counts)
    assert summary["filled_bins"] == 2
    assert summary["sig_bins"] == 1
    assert np.isclose(summary["count_weighted_sig_mean"], 100.0)
    assert summary["n_features"] == 4 * n


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")

# endregion