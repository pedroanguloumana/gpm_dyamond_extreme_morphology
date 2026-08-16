"""Binned rank correlation between two variables, on the same 2D grid as
``binned_threshold_delta``.

Standalone: run ``python -m src.binned_correlations`` from the repo root (or ``pytest``
on this file) to execute the tests at the bottom. Run it as ``python
src/binned_correlations.py`` instead and ``src/secrets.py`` shadows the standard
library's ``secrets``, which numpy.random imports.
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

try:  # importable as ``src.binned_correlations`` or run directly from ``src/``
    from ..src.binned_threshold_delta import _bin_groups, summarise_grid
except ImportError:  # pragma: no cover - only hit when run as a script
    from binned_threshold_delta import _bin_groups, summarise_grid


class BinnedCorrelation:
    """Rank correlation of ``data_a`` with ``data_b``, computed in 2D bins of
    ``(x_data, y_data)``.

    Rank correlation rather than Pearson because the variables here (feature area,
    max precip, rho) are heavy-tailed and related monotonically rather than linearly,
    and because ranks make the answer invariant to the log/linear choice of axis.

    Ties are handled by the mid-rank convention (Spearman) or by tau-b (Kendall), so
    the result does not depend on the order the rows arrived in -- the same property
    that motivated the threshold split in ``BinnedThresholdDelta``. For a variable as
    tied as rho = largest_core / n_conv, where ~70% of values sit at exactly 1.0,
    ``method="kendall"`` is the more honest choice: tau-b divides out the tied pairs
    instead of letting mid-ranks compress the coefficient toward zero.

    Bins with fewer than ``samp_min`` samples, or in which either variable is constant
    (so no correlation is defined), are left blank.

    The p-value comes from permuting one variable within the bin, so the null is
    "``data_a`` and ``data_b`` are independent given the bin".
    """

    def __init__(
        self,
        data_a,
        data_b,
        x_data,
        y_data,
        x_bins,
        y_bins,
        method: str = "spearman",
        samp_min: int = 20,
        n_perm: int = 200,
        seed: int = 67,
    ):
        """
        Args:
            data_a, data_b: the two variables to correlate.
            x_data, y_data: coordinates each sample is binned by.
            x_bins, y_bins: bin edges. The returned grids are (y, x), i.e. rows are
                ``y_bins`` and columns are ``x_bins``, matching pcolormesh.
            method: "spearman" (mid-rank Pearson) or "kendall" (tau-b).
            samp_min: minimum samples in a bin for it to be filled.
            n_perm: permutation draws for the p-value; 0 skips the test and leaves the
                p-grid as NaN. Kendall permutations are ~n log n each and are
                noticeably slower than Spearman's, which are a single matrix product.
            seed: permutation seed.
        """
        assert method in ("spearman", "kendall"), (
            f"method must be 'spearman' or 'kendall', got {method!r}"
        )
        assert samp_min >= 3, "samp_min must be at least 3 for a correlation to mean much"

        a = np.asarray(data_a, dtype=float)
        b = np.asarray(data_b, dtype=float)
        x = np.asarray(x_data, dtype=float)
        y = np.asarray(y_data, dtype=float)
        n = a.size
        assert b.size == n and x.size == n and y.size == n, (
            "data_a, data_b, x_data and y_data must be the same length"
        )

        # Drop unusable rows before binning, so counts are the number of features each
        # correlation actually rests on.
        good = np.isfinite(a) & np.isfinite(b) & np.isfinite(x) & np.isfinite(y)
        self.data_a = a[good]
        self.data_b = b[good]
        self.x_data = x[good]
        self.y_data = y[good]
        self.n_dropped = int((~good).sum())

        self.x_bins = np.asarray(x_bins, dtype=float)
        self.y_bins = np.asarray(y_bins, dtype=float)
        self.method = method
        self.samp_min = samp_min
        self.n_perm = n_perm
        self.seed = seed

        # Filled by compute(): samples actually used per bin, i.e. every sample in a
        # filled bin and 0 in a blank one. Use it as the plotting weight.
        self.used_counts = None

    @staticmethod
    def _array_midpoints(arr):
        """Bin midpoints from bin edges."""
        return (arr[:-1] + arr[1:]) / 2

    @staticmethod
    def _corr(a, b, method):
        """Rank correlation of one bin, or NaN if it is not defined.

        Spearman is Pearson on mid-ranks; Kendall is tau-b. Both are NaN when either
        variable is constant, since the coefficient then has a zero denominator.
        """
        if method == "kendall":
            r = stats.kendalltau(a, b).statistic
        else:
            ra, rb = stats.rankdata(a), stats.rankdata(b)
            r = _pearson(ra, rb)
        return float(r) if np.isfinite(r) else np.nan

    def _pval(self, a, b, obs, rng):
        """Two-sided permutation p-value for one bin.

        ``data_b`` is permuted against a fixed ``data_a``, which breaks any association
        between them while leaving both marginal distributions -- including their tie
        structure -- exactly as observed.

        The pairs are put in a canonical order first. The null distribution does not
        depend on that order, but the drawn permutations do, so sorting is what keeps
        the p-value from moving when the input rows are shuffled.
        """
        n = a.size
        if self.method == "kendall":
            order = np.lexsort((b, a))
            a, b = a[order], b[order]
            null = np.array([
                stats.kendalltau(a, rng.permutation(b)).statistic
                for _ in range(self.n_perm)
            ])
        else:
            # Permuting ranks is the same null as permuting values, and lets the whole
            # draw be one matrix product. Chunked so a large bin cannot blow up memory.
            ra = stats.rankdata(a)
            rb = stats.rankdata(b)
            order = np.lexsort((rb, ra))
            ra, rb = ra[order], rb[order]
            ra = ra - ra.mean()
            rb = rb - rb.mean()
            denom = np.linalg.norm(ra) * np.linalg.norm(rb)
            chunk = max(1, int(4e6 // n))
            null = np.concatenate([
                rng.permuted(np.tile(rb, (size, 1)), axis=1) @ ra / denom
                for size in _chunks(self.n_perm, chunk)
            ])
        null = null[np.isfinite(null)]
        return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1))

    def compute(self):
        """Return ``(s, p, counts)`` on the (y_bins, x_bins) grid.

        ``s`` is the rank correlation, ``p`` its permutation p-value, and ``counts`` the
        number of samples in each bin -- *every* sample, including those in bins left
        blank for being too small. ``self.used_counts`` holds the samples behind the
        filled bins only.
        """
        ny, nx = self.y_bins.size - 1, self.x_bins.size - 1
        groups, counts = _bin_groups(self.y_data, self.x_data, self.y_bins, self.x_bins)

        s = np.full(ny * nx, np.nan)
        p = np.full(ny * nx, np.nan)
        used = np.zeros(ny * nx)
        rng = np.random.default_rng(self.seed)

        for bin_index, idx in groups:
            if idx.size < self.samp_min:
                continue
            a, b = self.data_a[idx], self.data_b[idx]
            r = self._corr(a, b, self.method)
            if not np.isfinite(r):
                # One of the variables is constant in this bin: no correlation exists.
                continue

            used[bin_index] = idx.size
            s[bin_index] = r
            if self.n_perm:
                p[bin_index] = self._pval(a, b, r, rng)

        shape = (ny, nx)
        self.used_counts = used.reshape(shape)
        return s.reshape(shape), p.reshape(shape), counts

    @staticmethod
    def summarise(s, p, counts, alpha=0.05):
        """Summary statistics of a computed grid. See ``summarise_grid``."""
        return summarise_grid(s, p, counts, alpha=alpha)

    def plot(self, xlabel, ylabel, title, cmap, norm, xscale="log", ax=None,
             add_text=True, return_p=False, return_counts=False, weight_by_count=True,
             alpha=0.05):
        """Plot the binned correlation and report its summary.

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
        summary = self.summarise(s, p, self.used_counts, alpha=alpha)
        summary["n_features"] = float(np.nansum(counts))     # every sample in the grid
        summary["n_in_correlation"] = float(np.nansum(self.used_counts))
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
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale(xscale)
        ax.grid()

        symbol = r"$\tau$" if self.method == "kendall" else r"$r_s$"
        if add_text:
            label = "Wtd avg" if weight_by_count else "Avg"
            ax.text(
                0.95, 0.95,
                f"{label} {symbol}:{headline:.2f}",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=12,
                bbox=dict(facecolor="wheat", edgecolor="black",
                          boxstyle="round,pad=0.3"),
            )
        print(
            f"{title} count-weighted significant-bin mean {self.method}: "
            f"{summary['count_weighted_sig_mean']:.3f} "
            f"(unweighted: {summary['sig_mean']:.3f}, "
            f"{summary['sig_bins']}/{summary['filled_bins']} bins significant)"
        )

        out = [fig, ax, s, headline]
        if return_p:
            out.append(p)
        if return_counts:
            out.append(counts)
        return tuple(out)


def _pearson(a, b):
    """Pearson correlation, NaN when either input is constant."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return np.nan if denom == 0 else float(a @ b / denom)


def _chunks(total, size):
    """Split ``total`` into chunks of at most ``size``."""
    full, rest = divmod(total, size)
    return [size] * full + ([rest] if rest else [])


# region: TESTS

def _one_bin(a, b):
    """One bin's worth of samples, all landing at (x, y) = (5, 0.5)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = a.size
    return dict(data_a=a, data_b=b,
                x_data=np.full(n, 5.0), y_data=np.full(n, 0.5),
                x_bins=np.array([0.0, 10.0]), y_bins=np.array([0.0, 1.0]))


def test_perfect_monotonic_relation_gives_one():
    a = np.arange(30.0)
    for method in ("spearman", "kendall"):
        s, _, _ = BinnedCorrelation(**_one_bin(a, np.exp(a / 5)), method=method,
                                    samp_min=5, n_perm=0).compute()
        assert np.allclose(s, 1.0), method
        s, _, _ = BinnedCorrelation(**_one_bin(a, -a), method=method,
                                    samp_min=5, n_perm=0).compute()
        assert np.allclose(s, -1.0), method


def test_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = 0.4 * a + rng.normal(size=200)
    for method, ref in (("spearman", stats.spearmanr(a, b).statistic),
                        ("kendall", stats.kendalltau(a, b).statistic)):
        s, _, _ = BinnedCorrelation(**_one_bin(a, b), method=method,
                                    samp_min=5, n_perm=0).compute()
        assert np.isclose(s[0, 0], ref), method


def test_rank_correlation_is_scale_invariant():
    """Log or linear axis, same answer -- the reason for ranks over Pearson."""
    rng = np.random.default_rng(1)
    a = rng.lognormal(size=150)
    b = rng.lognormal(size=150)
    s0, _, _ = BinnedCorrelation(**_one_bin(a, b), samp_min=5, n_perm=0).compute()
    s1, _, _ = BinnedCorrelation(**_one_bin(np.log(a), b), samp_min=5,
                                 n_perm=0).compute()
    assert np.allclose(s0, s1)


def test_ties_handled_and_order_independent():
    """Heavy ties, as in rho: mid-ranks and tau-b both ignore row order."""
    rng = np.random.default_rng(2)
    n = 120
    a = np.where(rng.random(n) < 0.7, 1.0, rng.random(n))   # ~70% tied at 1.0
    b = rng.normal(size=n)
    perm = rng.permutation(n)
    for method in ("spearman", "kendall"):
        s0, p0, _ = BinnedCorrelation(**_one_bin(a, b), method=method, samp_min=5,
                                      n_perm=50).compute()
        s1, p1, _ = BinnedCorrelation(**_one_bin(a[perm], b[perm]), method=method,
                                      samp_min=5, n_perm=50).compute()
        assert np.array_equal(s0, s1, equal_nan=True), method
        assert np.array_equal(p0, p1, equal_nan=True), method


def test_constant_variable_blanks_the_bin():
    a = np.ones(40)
    b = np.arange(40.0)
    for method in ("spearman", "kendall"):
        s, p, counts = BinnedCorrelation(**_one_bin(a, b), method=method, samp_min=5,
                                         n_perm=0).compute()
        assert np.all(np.isnan(s)) and np.all(np.isnan(p)), method
        assert counts[0, 0] == 40   # the samples are still counted


def test_samp_min_blanks_the_bin():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=10), rng.normal(size=10)
    bc = BinnedCorrelation(**_one_bin(a, b), samp_min=20, n_perm=0)
    s, p, counts = bc.compute()
    assert np.all(np.isnan(s)) and np.all(np.isnan(p))
    assert counts[0, 0] == 10 and bc.used_counts[0, 0] == 0


def test_used_counts_track_filled_bins():
    rng = np.random.default_rng(4)
    n = 30
    a = np.r_[rng.normal(size=n), rng.normal(size=5)]      # second bin too small
    b = np.r_[rng.normal(size=n), rng.normal(size=5)]
    kw = _one_bin(a, b)
    kw["x_data"] = np.r_[np.full(n, 5.0), np.full(5, 15.0)]
    kw["x_bins"] = np.array([0.0, 10.0, 20.0])
    bc = BinnedCorrelation(**kw, samp_min=20, n_perm=0)
    _, _, counts = bc.compute()
    assert counts[0, 0] == n and counts[0, 1] == 5
    assert bc.used_counts[0, 0] == n and bc.used_counts[0, 1] == 0


def test_nonfinite_rows_dropped():
    a = np.arange(40.0)
    b = np.arange(40.0) ** 2
    kw = _one_bin(a, b)
    kw["data_a"] = kw["data_a"].copy()
    kw["data_a"][0] = np.nan
    kw["data_b"] = kw["data_b"].copy()
    kw["data_b"][-1] = np.inf
    bc = BinnedCorrelation(**kw, samp_min=5, n_perm=0)
    assert bc.n_dropped == 2
    _, _, counts = bc.compute()
    assert counts[0, 0] == 38


def test_grid_orientation_is_y_by_x():
    """Rows are y_bins, columns are x_bins -- the layout pcolormesh expects."""
    rng = np.random.default_rng(5)
    n = 40
    kw = _one_bin(rng.normal(size=n), rng.normal(size=n))
    kw["x_data"] = np.full(n, 25.0)   # third x bin
    kw["y_data"] = np.full(n, 0.15)   # second y bin
    kw["x_bins"] = np.array([0.0, 10.0, 20.0, 30.0])
    kw["y_bins"] = np.array([0.0, 0.1, 0.2])
    s, _, counts = BinnedCorrelation(**kw, samp_min=5, n_perm=0).compute()
    assert s.shape == (2, 3) and counts.shape == (2, 3)
    assert np.isfinite(s[1, 2])
    assert np.isnan(np.delete(s, 5)).all()   # every other bin is empty
    assert counts[1, 2] == n


def test_out_of_range_samples_ignored():
    rng = np.random.default_rng(6)
    n = 40
    kw = _one_bin(rng.normal(size=n), rng.normal(size=n))
    kw["x_data"] = kw["x_data"].copy()
    kw["x_data"][:5] = 999.0   # outside x_bins
    _, _, counts = BinnedCorrelation(**kw, samp_min=5, n_perm=0).compute()
    assert counts.sum() == n - 5


def test_independent_data_gives_large_p():
    rng = np.random.default_rng(7)
    n = 200
    s, p, _ = BinnedCorrelation(**_one_bin(rng.normal(size=n), rng.normal(size=n)),
                                samp_min=5, n_perm=400).compute()
    assert abs(s[0, 0]) < 0.2
    assert p[0, 0] > 0.05


def test_strong_relation_gives_minimum_p():
    a = np.arange(60.0)
    b = a + np.random.default_rng(8).normal(0, 0.1, 60)
    for method in ("spearman", "kendall"):
        _, p, _ = BinnedCorrelation(**_one_bin(a, b), method=method, samp_min=5,
                                    n_perm=200).compute()
        assert np.allclose(p, 1 / 201), method


def test_permutation_pvalue_matches_scipy_permutation_test():
    """Same null, so the two p-values should agree to permutation noise."""
    rng = np.random.default_rng(9)
    n = 80
    a = rng.normal(size=n)
    b = 0.3 * a + rng.normal(size=n)
    _, p, _ = BinnedCorrelation(**_one_bin(a, b), samp_min=5, n_perm=2000,
                                seed=11).compute()
    ref = stats.permutation_test(
        (b,), lambda z: stats.spearmanr(a, z).statistic,
        permutation_type="pairings", n_resamples=2000, alternative="two-sided",
        random_state=12,
    ).pvalue
    assert abs(p[0, 0] - ref) < 0.03


def test_pvalue_is_reproducible():
    rng = np.random.default_rng(10)
    kw = _one_bin(rng.normal(size=60), rng.normal(size=60))
    a = BinnedCorrelation(**kw, samp_min=5, n_perm=100, seed=5).compute()[1]
    b = BinnedCorrelation(**kw, samp_min=5, n_perm=100, seed=5).compute()[1]
    assert np.array_equal(a, b, equal_nan=True)


def test_chunking_covers_every_permutation():
    """The memory cap on the permutation matrix must not drop or add draws."""
    for total, size in ((200, 64), (200, 200), (200, 1000), (5, 2), (0, 10)):
        chunks = _chunks(total, size)
        assert sum(chunks) == total, (total, size)
        assert all(0 < c <= size for c in chunks), (total, size)


def test_chunked_permutations_give_the_same_p():
    """A bin large enough to be chunked still gets a sane p-value."""
    rng = np.random.default_rng(11)
    n = 3000                      # chunk = 4e6 // 3000 = 1333 < n_perm
    a = np.arange(n, dtype=float)
    b = a + rng.normal(0, 1.0, n)
    s, p, _ = BinnedCorrelation(**_one_bin(a, b), samp_min=5, n_perm=2000,
                                seed=3).compute()
    assert s[0, 0] > 0.99
    assert np.allclose(p, 1 / 2001)


def test_n_perm_zero_skips_the_test():
    rng = np.random.default_rng(12)
    s, p, _ = BinnedCorrelation(**_one_bin(rng.normal(size=40), rng.normal(size=40)),
                                samp_min=5, n_perm=0).compute()
    assert np.all(np.isfinite(s)) and np.all(np.isnan(p))


def test_bad_method_rejected():
    kw = _one_bin(np.arange(10.0), np.arange(10.0))
    try:
        BinnedCorrelation(**kw, method="pearson")
    except AssertionError:
        return
    raise AssertionError("expected an unknown method to be rejected")


def test_mismatched_lengths_rejected():
    kw = _one_bin(np.arange(10.0), np.arange(10.0))
    kw["data_b"] = kw["data_b"][:-1]
    try:
        BinnedCorrelation(**kw)
    except AssertionError:
        return
    raise AssertionError("expected mismatched input lengths to be rejected")


def test_summarise_reports_significant_bins():
    """Two bins, one with a real relation and one without."""
    rng = np.random.default_rng(13)
    n = 60
    a = np.r_[np.arange(n, dtype=float), rng.normal(size=n)]
    b = np.r_[np.arange(n, dtype=float) + rng.normal(0, 0.1, n), rng.normal(size=n)]
    x = np.r_[np.full(n, 5.0), np.full(n, 15.0)]
    bc = BinnedCorrelation(
        data_a=a, data_b=b, x_data=x, y_data=np.full(2 * n, 0.5),
        x_bins=np.array([0.0, 10.0, 20.0]), y_bins=np.array([0.0, 1.0]),
        samp_min=5, n_perm=200,
    )
    s, p, counts = bc.compute()
    summary = bc.summarise(s, p, counts)
    assert summary["filled_bins"] == 2
    assert summary["sig_bins"] == 1
    assert np.isclose(summary["count_weighted_sig_mean"], s[0, 0])
    assert summary["n_features"] == 2 * n


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")

# endregion
