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


class BinnedQuantileDelta:
    """Compute and plot binned quantile differences on a 2D grid.

    In each (x, y) bin the sample is split into a "low" and a "high" group by
    ``data_to_sort``, and the difference of the mean ``data_to_diff`` between the two
    groups is reported. Two ways of making that split are available:

    ``split="quantile"`` (the original behaviour)
        Sort by ``data_to_sort`` and take the ``low_quantile`` / ``high_quantile``
        slices, e.g. the bottom and top deciles. This is fragile when the sorting
        variable is heavily tied: for the concentration rho = largest_core / n_conv,
        ~70% of features sit at exactly 1.0, so which tied feature lands in the top
        decile and which in the bottom is decided by row order alone, moving the
        answer by tens of percent. Pass ``tie_breaker`` to make that choice
        deterministic and independent of row order.

    ``split="threshold"``
        Group by value instead of by rank: low is ``data_to_sort <= low_threshold``,
        high is ``data_to_sort >= high_threshold``. Ties cannot straddle a group
        boundary, so the result does not depend on row order at all. Bins where either
        group has fewer than ``group_min`` members are left blank rather than being
        filled with a contrast the data cannot support.
    """

    @staticmethod
    def _pack_columns(*cols):
        """Pack columns into an object array of tuples.

        Needed because scipy.binned_statistic_2d only accepts 1D values
        arrays. Packing the columns as tuples in an object array lets us pass
        the sorting, diffing and tie-breaking data through a single argument.
        """
        cols = [np.asarray(c, dtype=float) for c in cols]
        n = len(cols[0])
        assert all(len(c) == n for c in cols)
        arr = np.empty(n, dtype=object)
        for i, row in enumerate(zip(*cols)):
            arr[i] = row
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
        tie_breaker=None,
        split: str = "quantile",
        low_threshold=None,
        high_threshold=None,
        group_min=None,
    ):
        """
        Args:
            data_to_sort: variable the low/high split is made on, e.g. rho.
            data_to_diff: variable differenced between the two groups, e.g. max precip.
            x_data, y_data: coordinates each sample is binned by.
            x_bins, y_bins: bin edges.
            low_quantile, high_quantile: (start, end) quantile bounds of the two groups,
                used by ``split="quantile"`` only.
            percent_diff: report the difference as a percentage of the low-group mean.
            samp_min: minimum number of samples in a bin for it to be filled at all.
            n_perm: permutation-test draws used for the p-value; 0 skips the test.
            tie_breaker: optional secondary sort key, same length as ``data_to_sort``.
                Ties in ``data_to_sort`` are ordered by this key instead of by row
                order, which is what makes ``split="quantile"`` reproducible when the
                sorting variable is heavily tied. Ignored by ``split="threshold"``.
            split: "quantile" (rank-based groups) or "threshold" (value-based groups).
            low_threshold, high_threshold: group bounds for ``split="threshold"``; low
                is ``<= low_threshold`` and high is ``>= high_threshold``. Required for
                that split, ignored otherwise.
            group_min: minimum members in each of the two groups for a bin to be
                filled. Defaults to 5 for ``split="threshold"``, where the group sizes
                are set by the data rather than by the quantile widths, and to 1 for
                ``split="quantile"``, where the bin-level ``samp_min`` already fixes
                them.
        """
        assert split in ("quantile", "threshold"), f"Unknown split {split!r}"
        if split == "threshold":
            assert low_threshold is not None and high_threshold is not None, (
                "split='threshold' needs low_threshold and high_threshold"
            )
            assert low_threshold < high_threshold, (
                "low_threshold must be < high_threshold so the two groups are disjoint"
            )

        if tie_breaker is None:
            self.data_array = self._pack_columns(data_to_sort, data_to_diff)
        else:
            self.data_array = self._pack_columns(data_to_sort, data_to_diff, tie_breaker)
        self.has_tie_breaker = tie_breaker is not None
        self.x_data = x_data
        self.y_data = y_data
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.percent_diff = percent_diff
        self.samp_min = samp_min
        self.n_perm = n_perm
        self.split = split
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        if group_min is None:
            group_min = 5 if split == "threshold" else 1
        self.group_min = group_min

    def _quantile_indices(self, n):
        """Convert quantile bounds to array indices for a sample of size n."""
        low_start = round(n * self.low_quantile[0])
        low_end = round(n * self.low_quantile[1])
        high_start = round(n * self.high_quantile[0])
        high_end = round(n * self.high_quantile[1])
        return low_start, low_end, high_start, high_end

    def _sorted(self, arr):
        """Sort a bin's samples by data_to_sort, breaking ties by the secondary key.

        Without a tie breaker this is the original, order-dependent sort: ties are
        resolved by whatever order the rows happen to be stored in, which for a heavily
        tied sorting variable moves the answer by tens of percent. Kept as the default
        only so the published numbers stay reproducible.
        """
        if self.has_tie_breaker:
            order = np.lexsort((arr[:, 2], arr[:, 0]))
        else:
            order = arr[:, 0].argsort()
        return arr[order]

    @staticmethod
    def _canonical(arr):
        """Row-order-independent ordering of a bin's samples.

        The permutation test draws from a fixed seed, so it only gives the same answer
        twice if the sample it shuffles is in the same order both times.
        """
        return arr[np.lexsort(tuple(arr[:, i] for i in range(arr.shape[1] - 1, -1, -1)))]

    def _groups(self, arr):
        """Return the (low, high) diff values of a bin, or None if it cannot be split."""
        if arr.shape[0] < self.samp_min:
            return None

        if self.split == "threshold":
            low = arr[arr[:, 0] <= self.low_threshold, 1]
            high = arr[arr[:, 0] >= self.high_threshold, 1]
            if low.size < self.group_min or high.size < self.group_min:
                return None
            return low, high

        sorted_arr = self._sorted(arr)
        li0, li1, hi0, hi1 = self._quantile_indices(arr.shape[0])
        low, high = sorted_arr[li0:li1, 1], sorted_arr[hi0:hi1, 1]
        if low.size < self.group_min or high.size < self.group_min:
            return None
        return low, high

    def binned_quantile_delta(self, tuple_arr, percent_diff=None):
        if percent_diff is None:
            percent_diff = self.percent_diff

        groups = self._groups(np.array(tuple_arr))
        if groups is None:
            return np.nan
        low, high = groups

        delta = np.nanmean(high) - np.nanmean(low)
        if percent_diff:
            delta = 100 * delta / np.nanmean(low)
        return delta

    def binned_quantile_delta_pval(self, tuple_arr):
        arr = np.array(tuple_arr)
        groups = self._groups(arr)
        if groups is None:
            return np.nan

        delta = self.binned_quantile_delta(tuple_arr, percent_diff=False)

        if self.split == "threshold":
            # The two groups are disjoint by construction, so take them off either end
            # of the shuffled sample at the sizes the data actually gave.
            n_low, n_high = groups[0].size, groups[1].size
            low_sl, high_sl = slice(0, n_low), slice(arr.shape[0] - n_high, None)
            arr = self._canonical(arr)
        else:
            li0, li1, hi0, hi1 = self._quantile_indices(arr.shape[0])
            low_sl, high_sl = slice(li0, li1), slice(hi0, hi1)

        # Shuffle the sample and re-split it at the same group sizes, so the null is
        # "data_to_diff carries no information about data_to_sort".
        rng = np.random.default_rng(seed=67)
        random_deltas = np.empty(self.n_perm)
        for i in range(self.n_perm):
            perm = rng.permutation(arr)
            random_deltas[i] = (
                np.nanmean(perm[high_sl, 1]) - np.nanmean(perm[low_sl, 1])
            )

        p = (np.sum(np.abs(random_deltas) >= np.abs(delta)) + 1) / (self.n_perm + 1)
        return p

    def compute(self):
        """Return (s, p, counts) on the (y_bins, x_bins) grid.

        ``counts`` is the number of samples in each bin, i.e. the weights the
        count-weighted summary uses.
        """
        s = stats.binned_statistic_2d(
            self.y_data, self.x_data, self.data_array,
            statistic=self.binned_quantile_delta,
            bins=[self.y_bins, self.x_bins],
        ).statistic.astype(float)

        p = stats.binned_statistic_2d(
            self.y_data, self.x_data, self.data_array,
            statistic=self.binned_quantile_delta_pval,
            bins=[self.y_bins, self.x_bins],
        ).statistic.astype(float)

        counts = stats.binned_statistic_2d(
            self.y_data, self.x_data, None,
            statistic='count',
            bins=[self.y_bins, self.x_bins],
        ).statistic.astype(float)

        return s, p, counts

    @staticmethod
    def summarise(s, p, counts, alpha=0.05):
        """Summary statistics of a computed grid. See ``summarise_grid``."""
        return summarise_grid(s, p, counts, alpha=alpha)

    def plot(self, xlabel, ylabel, title, cmap, norm, xscale='log', ax=None,
             add_text=True, return_p=False, return_counts=False, weight_by_count=True,
             alpha=0.05):
        """Plot the binned delta and report its summary.

        Args:
            weight_by_count: report the count-weighted mean over significant bins as
                the headline number (annotation and returned scalar). This is the
                recommended statistic; set False for the old unweighted significant-bin
                mean. Both are printed either way.
            return_p: also return the p-value grid.
            return_counts: also return the per-bin counts.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        s, p, counts = self.compute()
        summary = self.summarise(s, p, counts, alpha=alpha)
        headline = (
            summary["count_weighted_sig_mean"] if weight_by_count else summary["sig_mean"]
        )

        # Plot heatmap
        x_mesh, y_mesh = np.meshgrid(self.x_bins, self.y_bins)
        im = ax.pcolormesh(x_mesh, y_mesh, s, cmap=cmap, norm=norm)
        fig.colorbar(im)

        # Hatch significant bins
        xx, yy = np.meshgrid(self._array_midpoints(self.x_bins), self._array_midpoints(self.y_bins))
        ax.contourf(
            xx, yy, (p <= alpha).astype(float),
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

        if add_text:
            label = 'Wtd avg' if weight_by_count else 'Avg'
            ax.text(
                0.95, 0.95,
                f'{label}:{headline:.2f}%',
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=12,
                bbox=dict(facecolor='wheat', edgecolor='black', boxstyle='round,pad=0.3',)
            )
        print(
            f'{title} count-weighted significant-bin mean: '
            f'{summary["count_weighted_sig_mean"]:.3f} '
            f'(unweighted: {summary["sig_mean"]:.3f}, '
            f'{summary["sig_bins"]}/{summary["filled_bins"]} bins significant)'
        )

        out = [fig, ax, s, headline]
        if return_p:
            out.append(p)
        if return_counts:
            out.append(counts)
        return tuple(out)


class BinnedTLSSlope:
    """Regress log(response) onto a predictor by total least squares, in 2D bins.

    Same grid as :class:`BinnedQuantileDelta`, but each (x, y) bin is collapsed to a
    *slope* rather than to a two-group difference. For the concentration study the
    predictor is rho = largest_core / n_conv and the response is max precip, so each
    bin reports how much MaxPr changes per unit of rho at fixed area and fixed
    convective area fraction.

    **Why log.** The fit is on ``log(response)``, so the slope is a fractional
    sensitivity rather than a mm/hr one. With ``percent=True`` it is reported as
    ``100 * (exp(b) - 1)``, the percentage change in MaxPr per unit rho, which is
    directly comparable with the percentage deltas of :class:`BinnedQuantileDelta`
    instead of being a separate quantity in different units. Logging also pulls the
    heavily skewed MaxPr distribution towards normality, which is what the analytic
    p-value assumes.

    **Why total least squares.** OLS minimises vertical residuals and treats the
    predictor as known exactly. rho is a ratio of pixel counts, i.e. measured with
    error, so the OLS slope is attenuated towards zero by roughly
    ``1 / (1 + sigma_err^2 / sigma_true^2)``. TLS minimises perpendicular distance
    and removes that attenuation. The forward and reverse OLS slopes are computed
    alongside and kept in ``ols_grid`` and ``ols_reverse_grid``; the true slope lies
    between them, so the size of the attenuation is visible rather than assumed.

    **Scaling matters.** TLS is not scale invariant: "perpendicular" mixes the units
    of the two axes, so the answer changes if the response is rescaled. The
    ``scaling`` argument fixes this:

    ``"standardize"`` (default)
        Standardise both variables within the bin, then fit orthogonally. This is
        reduced-major-axis regression, and the slope comes out as
        ``sign(r) * s_y / s_x``. It needs no assumption about the error variances,
        but note the magnitude does *not* depend on ``|r|``: a bin with no
        relationship still returns a slope of that size, only its sign is noise.
        That is why significance here is tested on the correlation, not on the
        slope (see below), and why ``r_grid`` should be inspected alongside the map.
        The RMA slope is ``b_ols / |r|``, the geometric mean of the forward and
        reverse OLS fits, so it sits at the geometric midpoint of the bracket
        ``[ols_grid, ols_reverse_grid]`` that the true slope must lie inside. Those
        two grids are the honest error bar on the map, and are worth plotting when
        the number is quoted.

    ``"attenuation"`` (recommended for rho)
        The method-of-moments errors-in-variables estimator,
        ``b = s_xy / (s_xx - sigma_x_err^2) = b_ols / reliability``. It needs only
        the error in the *predictor*, which for rho is estimable rather than
        assumed: rho is a ratio of pixel counts, so
        ``sigma_rho ~ sqrt(rho (1 - rho) / n_conv)``. Pass that as ``predictor_err``.
        Nothing about the response error has to be guessed, because response noise
        does not bias an OLS slope in the first place -- only predictor noise does,
        and this undoes exactly that. It is also numerically well behaved: as the
        covariance goes to zero the slope goes to zero rather than diverging.

    ``"deming"``
        Orthogonal fit after scaling the response by ``1 / sqrt(delta)``, where
        ``delta`` is the ratio of the error variance in log(response) to the error
        variance in the predictor, either fixed or set per bin from
        ``predictor_err`` and ``response_err`` (``delta_grid`` records what was
        used). Unbiased when delta is right, but it needs a response error nobody
        can measure here, and its slope diverges as the covariance goes to zero.
        Prefer ``"attenuation"`` unless the response error is genuinely known.

    The distinction that decides between them is that errors-in-variables
    correction is meant to undo *measurement* error only. MaxPr varies between
    features at fixed area, CAF and rho for real physical reasons too, and that
    equation error should not be corrected away. RMA corrects for all of the
    scatter regardless of its origin, so it over-corrects by ``1 / |r|`` whenever
    the response carries genuine spread -- which in these bins it does. Deming with
    an honest error ratio corrects for only the measurement part.

    ``"none"``
        Plain orthogonal fit in raw units. Kept for completeness; the units of the
        two axes are mixed, so it is only meaningful if they are comparable already.

    **Significance.** The p-value tests the correlation, not the slope: the null is
    "log(response) carries no linear information about the predictor in this bin".
    Testing the slope magnitude would be vacuous under ``scaling="standardize"``,
    where permuting the response leaves ``s_y / s_x`` — and hence ``|slope|`` —
    exactly unchanged, so every bin would come out insignificant. ``p_method``
    selects the analytic Pearson t-test (default, fast) or a permutation test on
    ``|r|``; the two agree closely once the response is logged.
    """

    def __init__(
        self,
        predictor,
        response,
        x_data,
        y_data,
        x_bins,
        y_bins,
        log_response: bool = True,
        percent: bool = True,
        scaling: str = "standardize",
        delta: float = 1.0,
        predictor_err=None,
        response_err=None,
        samp_min: int = 10,
        min_distinct: int = 3,
        min_off_mode: int = 5,
        min_reliability: float = 0.1,
        min_abs_r: float = 0.0,
        p_method: str = "analytic",
        n_perm: int = 200,
        weights=None,
        seed: int = 67,
    ):
        """
        Args:
            predictor: regressed onto, e.g. rho. Measured with error, hence TLS.
            response: regressed, e.g. max precip in mm/hr. Logged unless
                ``log_response=False``; non-positive values are dropped when logging.
            x_data, y_data: coordinates each feature is binned by, e.g. area and
                convective area fraction.
            x_bins, y_bins: bin edges.
            log_response: fit log(response) rather than response.
            percent: report the slope as ``100 * (exp(b) - 1)``, the percentage change
                in the response per unit predictor. Requires ``log_response``.
            scaling: "standardize", "deming" or "none"; see the class docstring.
            delta: fixed error variance ratio (response over predictor) for
                ``scaling="deming"``. Pass None to set it per bin from
                ``predictor_err`` and ``response_err`` instead. Ignored otherwise.
            predictor_err: per-feature 1-sigma measurement error on the predictor.
                For rho this is the counting error of a ratio of pixel counts,
                ``sqrt(rho * (1 - rho) / px_ge_10mmhr)``.
            response_err: 1-sigma measurement error on the response, in the units
                being fitted -- log units when ``log_response``, where a fractional
                retrieval uncertainty of x translates to roughly x. Scalar or
                per-feature. Only the ratio to ``predictor_err`` matters.
            samp_min: minimum features in a bin for it to be fitted at all.
            min_distinct: minimum number of distinct predictor values in a bin. A bin
                where every feature has the same rho carries no slope, and one with
                two distinct values fits a line through two points.
            min_off_mode: minimum number of features whose predictor differs from the
                bin's modal value. rho is heavily tied at 1.0, so a bin can clear
                ``samp_min`` while the fit rests on two or three off-mode features;
                ``tie_frac_grid`` records how much of each bin sits at the mode.
            min_reliability: minimum fraction of the observed predictor variance that
                is real signal rather than measurement error,
                ``1 - mean(predictor_err^2) / s_xx``. Below this the errors-in-
                variables correction is not identifiable -- the data do not contain
                enough spread in the predictor to tell the true slope from the noise
                -- and the bin is blanked rather than returned as a huge number.
                Only applied when ``predictor_err`` is given; ``reliability_grid``
                records the value. This bites hard on rho: the median GPM feature has
                two convective pixels, so its rho is either 0.5 or 1 and carries
                about one bit. Restricting to features with enough convective pixels
                for rho to be resolved is the way to raise it.
            min_abs_r: minimum ``|r|`` for a bin to be fitted. Every TLS variant
                divides by the covariance, so all of them are unstable as ``r``
                approaches zero; under ``scaling="deming"`` the slope diverges
                outright. 0 leaves the screening to the p-value.
            p_method: "analytic" (Pearson t-test) or "permutation" (on ``|r|``).
            n_perm: permutation draws, used by ``p_method="permutation"`` only.
            weights: optional per-feature weights, e.g. ``px_ge_10mmhr`` to
                down-weight features whose rho is coarsely resolved. Applied as
                reliability weights to the covariance. This is an approximation: it
                assumes the error covariance of both variables scales as ``1 / w``,
                whereas only the predictor's counting error really does. Full
                per-point errors-in-variables would need ``scipy.odr``.
            seed: permutation-test seed.
        """
        assert scaling in ("standardize", "attenuation", "deming", "none"), (
            f"Unknown scaling {scaling!r}"
        )
        if scaling == "attenuation":
            assert predictor_err is not None, (
                "scaling='attenuation' corrects for the predictor's measurement error, "
                "so it needs predictor_err"
            )
        assert p_method in ("analytic", "permutation"), f"Unknown p_method {p_method!r}"
        assert not (percent and not log_response), (
            "percent=True reports exp(slope) - 1, which is only a percentage change "
            "if the fit is on the log of the response"
        )
        if scaling == "deming":
            if delta is None:
                assert predictor_err is not None and response_err is not None, (
                    "scaling='deming' with delta=None needs predictor_err and "
                    "response_err to set the variance ratio from the data"
                )
            else:
                assert delta > 0, "delta is a variance ratio, so it must be positive"

        predictor = np.asarray(predictor, dtype=float)
        response = np.asarray(response, dtype=float)
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        w = None if weights is None else np.asarray(weights, dtype=float)
        ex = None if predictor_err is None else np.asarray(predictor_err, dtype=float)
        ey = (None if response_err is None
              else np.broadcast_to(np.asarray(response_err, dtype=float),
                                   predictor.shape).astype(float))

        # Drop what cannot be fitted before binning, so the counts grid is the number
        # of features each slope actually rests on and can be used as its weight.
        good = (
            np.isfinite(predictor) & np.isfinite(response)
            & np.isfinite(x_data) & np.isfinite(y_data)
        )
        if log_response:
            good &= response > 0
        if w is not None:
            good &= np.isfinite(w) & (w > 0)
        if ex is not None:
            good &= np.isfinite(ex) & (ex > 0)
        if ey is not None:
            good &= np.isfinite(ey) & (ey > 0)

        self.predictor = predictor[good]
        self.response = np.log(response[good]) if log_response else response[good]
        self.x_data = x_data[good]
        self.y_data = y_data[good]
        self.weights = None if w is None else w[good]
        self.predictor_err = None if ex is None else ex[good]
        self.response_err = None if ey is None else ey[good]
        self.n_dropped = int((~good).sum())

        self.x_bins = np.asarray(x_bins, dtype=float)
        self.y_bins = np.asarray(y_bins, dtype=float)
        self.log_response = log_response
        self.percent = percent
        self.scaling = scaling
        self.delta = delta
        self.samp_min = samp_min
        self.min_distinct = min_distinct
        self.min_off_mode = min_off_mode
        self.min_reliability = min_reliability
        self.min_abs_r = min_abs_r
        self.p_method = p_method
        self.n_perm = n_perm
        self.seed = seed

        # Filled by compute(); kept as attributes because compute() returns only the
        # (slope, p, counts) triple that BinnedQuantileDelta's callers expect.
        self.r_grid = None
        self.ols_grid = None
        self.ols_reverse_grid = None
        self.intercept_grid = None
        self.tie_frac_grid = None
        self.delta_grid = None
        self.reliability_grid = None

    @staticmethod
    def _array_midpoints(arr):
        """Compute bin midpoints from bin edges."""
        return (arr[:-1] + arr[1:]) / 2

    @staticmethod
    def _orthogonal_slope(sxx, sxy, syy):
        """Slope of the major axis of a 2x2 covariance, i.e. the orthogonal TLS fit.

        Taken as the leading eigenvector rather than from the closed form, which
        divides by ``sxy`` and so blows up exactly where the fit is least defined. A
        vertical major axis (no spread in x) has no slope and returns NaN.
        """
        _, vecs = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
        vx, vy = vecs[:, -1]  # eigh sorts ascending, so the last column is the major axis
        if abs(vx) < 1e-12:
            return np.nan
        return vy / vx

    def _fit(self, x, y, w, ex=None, ey=None):
        """Fit one bin.

        Returns ``(slope, r, ols, ols_reverse, intercept, tie_frac, delta,
        reliability)``. Slopes are in the units the class reports, i.e. percent per
        unit predictor when ``percent`` is set. Everything but ``tie_frac`` and
        ``reliability`` is NaN when the bin cannot be fitted; those two are returned
        whenever they can be computed, because they are the diagnostics that explain
        most of the blanks.
        """
        n = x.size
        if n < max(self.samp_min, 3):
            return (np.nan,) * 8

        values, value_counts = np.unique(x, return_counts=True)
        tie_frac = value_counts.max() / n
        blank = (np.nan,) * 5 + (tie_frac, np.nan, np.nan)
        if values.size < self.min_distinct or n - value_counts.max() < self.min_off_mode:
            return blank

        cov = np.cov(np.vstack([x, y]), aweights=w)
        sxx, sxy, syy = cov[0, 0], cov[0, 1], cov[1, 1]
        if not (sxx > 0 and syy > 0):
            return blank

        # How much of the observed spread in the predictor is real rather than
        # measurement noise. Errors-in-variables correction is only identifiable
        # where this is comfortably positive.
        reliability = np.nan
        if ex is not None:
            reliability = 1 - np.mean(ex ** 2) / sxx
            blank = (np.nan,) * 5 + (tie_frac, np.nan, reliability)
            if not reliability > self.min_reliability:
                return blank

        r = sxy / np.sqrt(sxx * syy)
        if not abs(r) > self.min_abs_r:
            return blank
        ols = sxy / sxx                       # regression of y on x: attenuated
        ols_rev = syy / sxy if sxy != 0 else np.nan  # 1 / (regression of x on y)

        # Method of moments: divide out the share of the predictor's spread that is
        # noise. Not an orthogonal fit at all, so it skips the machinery below.
        delta = np.nan
        if self.scaling == "attenuation":
            slope = ols / reliability
            if w is None:
                intercept = y.mean() - slope * x.mean()
            else:
                intercept = np.average(y, weights=w) - slope * np.average(x, weights=w)
            if self.percent:
                with np.errstate(over='ignore'):
                    slope, ols, ols_rev = (100 * np.expm1(v) for v in (slope, ols, ols_rev))
            return slope, r, ols, ols_rev, intercept, tie_frac, delta, reliability

        # Scale both axes, fit orthogonally in the scaled space, then undo the scaling:
        # a fit on (x / fx, y / fy) has slope b', and b = b' * fy / fx.
        if self.scaling == "standardize":
            # In standardised space the covariance is [[1, r], [r, 1]], whose major
            # axis is degenerate at r = 0 -- the fit has no direction to report there.
            if abs(r) < 1e-12:
                return blank
            fx, fy = np.sqrt(sxx), np.sqrt(syy)
        elif self.scaling == "deming":
            if self.delta is not None:
                delta = self.delta
            else:
                # Mean error variances over the bin: delta is a ratio of variances,
                # so the per-feature errors enter only through their mean square.
                mx = np.mean(ex ** 2)
                if not mx > 0:
                    return blank
                delta = np.mean(ey ** 2) / mx
            fx, fy = 1.0, np.sqrt(delta)
        else:
            fx, fy = 1.0, 1.0

        slope = self._orthogonal_slope(sxx / fx**2, sxy / (fx * fy), syy / fy**2)
        if not np.isfinite(slope):
            return blank
        slope *= fy / fx

        if w is None:
            intercept = y.mean() - slope * x.mean()
        else:
            intercept = np.average(y, weights=w) - slope * np.average(x, weights=w)

        if self.percent:
            # expm1 overflows for the very large log-slopes a nearly vertical fit can
            # produce. The guards above should keep those out; let any survivor come
            # through as inf rather than as a warning.
            with np.errstate(over='ignore'):
                slope = 100 * np.expm1(slope)
                ols = 100 * np.expm1(ols)
                ols_rev = 100 * np.expm1(ols_rev)

        return slope, r, ols, ols_rev, intercept, tie_frac, delta, reliability

    def _pval(self, x, y, r, rng):
        """p-value for the null that the response is unrelated to the predictor.

        Tests the correlation rather than the slope: under ``scaling="standardize"``
        the slope magnitude is ``s_y / s_x``, which a permutation of y leaves exactly
        unchanged, so a slope-based test could never reject.
        """
        n = x.size
        if not np.isfinite(r) or n < 3:
            return np.nan
        if abs(r) >= 1:
            return 0.0

        if self.p_method == "analytic":
            t = r * np.sqrt((n - 2) / (1 - r**2))
            return float(2 * stats.t.sf(abs(t), n - 2))

        xc = x - x.mean()
        sx = np.sqrt(np.sum(xc**2))
        null = np.empty(self.n_perm)
        for i in range(self.n_perm):
            yp = rng.permutation(y)
            yc = yp - yp.mean()
            null[i] = abs(np.dot(xc, yc)) / (sx * np.sqrt(np.sum(yc**2)))
        return float((np.sum(null >= abs(r)) + 1) / (self.n_perm + 1))

    def compute(self):
        """Return (slope, p, counts) on the (y_bins, x_bins) grid.

        Also fills ``r_grid``, ``ols_grid``, ``ols_reverse_grid``, ``intercept_grid``,
        ``tie_frac_grid``, ``delta_grid`` and ``reliability_grid``. ``counts`` is the
        number of features in each bin, i.e. the weights the count-weighted summary
        uses -- note it counts every feature that fell in the bin, including those in
        bins the guards blanked.
        """
        ny, nx = self.y_bins.size - 1, self.x_bins.size - 1

        # Bin through scipy so the grid matches BinnedQuantileDelta's exactly, then do
        # the fits in one pass over the groups rather than one pass per output grid.
        binned = stats.binned_statistic_2d(
            self.y_data, self.x_data, None,
            statistic='count',
            bins=[self.y_bins, self.x_bins],
            expand_binnumbers=True,
        )
        counts = binned.statistic.astype(float)
        iy, ix = binned.binnumber
        in_range = (iy >= 1) & (iy <= ny) & (ix >= 1) & (ix <= nx)

        flat = np.where(in_range, (iy - 1) * nx + (ix - 1), -1)
        rows = np.flatnonzero(in_range)
        order = rows[np.argsort(flat[rows], kind='stable')]
        bin_of_row = flat[order]
        starts = np.flatnonzero(np.r_[True, np.diff(bin_of_row) != 0])
        group_bins = bin_of_row[starts]
        group_ends = np.r_[starts[1:], bin_of_row.size]

        grids = {k: np.full(ny * nx, np.nan) for k in
                 ('slope', 'p', 'r', 'ols', 'ols_rev', 'intercept', 'tie_frac', 'delta',
                  'reliability')}
        rng = np.random.default_rng(self.seed)

        for b, s0, s1 in zip(group_bins, starts, group_ends):
            idx = order[s0:s1]
            x, y = self.predictor[idx], self.response[idx]
            w = None if self.weights is None else self.weights[idx]
            ex = None if self.predictor_err is None else self.predictor_err[idx]
            ey = None if self.response_err is None else self.response_err[idx]
            slope, r, ols, ols_rev, intercept, tie_frac, delta, rel = self._fit(x, y, w, ex, ey)
            grids['tie_frac'][b] = tie_frac
            grids['reliability'][b] = rel
            if not np.isfinite(slope):
                continue
            grids['slope'][b] = slope
            grids['r'][b] = r
            grids['ols'][b] = ols
            grids['ols_rev'][b] = ols_rev
            grids['intercept'][b] = intercept
            grids['delta'][b] = delta
            grids['p'][b] = self._pval(x, y, r, rng)

        shape = (ny, nx)
        self.r_grid = grids['r'].reshape(shape)
        self.ols_grid = grids['ols'].reshape(shape)
        self.ols_reverse_grid = grids['ols_rev'].reshape(shape)
        self.intercept_grid = grids['intercept'].reshape(shape)
        self.tie_frac_grid = grids['tie_frac'].reshape(shape)
        self.delta_grid = grids['delta'].reshape(shape)
        self.reliability_grid = grids['reliability'].reshape(shape)

        return grids['slope'].reshape(shape), grids['p'].reshape(shape), counts

    @staticmethod
    def summarise(s, p, counts, alpha=0.05):
        """Summary statistics of a computed grid. See ``summarise_grid``."""
        return summarise_grid(s, p, counts, alpha=alpha)

    def plot(self, xlabel, ylabel, title, cmap, norm, xscale='log', ax=None,
             add_text=True, return_p=False, return_counts=False, weight_by_count=True,
             alpha=0.05):
        """Plot the binned slope and report its summary.

        Mirrors :meth:`BinnedQuantileDelta.plot`: significant bins are hatched, and the
        headline is the count-weighted mean over them.

        Args:
            weight_by_count: report the count-weighted mean over significant bins as
                the headline number. This is the recommended statistic; set False for
                the unweighted significant-bin mean. Both are printed either way.
            return_p: also return the p-value grid.
            return_counts: also return the per-bin counts.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        s, p, counts = self.compute()
        summary = self.summarise(s, p, counts, alpha=alpha)
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
            colors='none',
            hatches=['...'],
            linewidths=0,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale(xscale)
        ax.grid()

        unit = '%' if self.percent else ''
        if add_text:
            label = 'Wtd avg' if weight_by_count else 'Avg'
            ax.text(
                0.95, 0.95,
                f'{label}:{headline:.2f}{unit}',
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=12,
                bbox=dict(facecolor='wheat', edgecolor='black', boxstyle='round,pad=0.3',)
            )
        # The median |r| says how much of the map is a real relationship rather than
        # the s_y / s_x floor the standardised fit returns whatever the scatter.
        med_abs_r = np.nanmedian(np.abs(self.r_grid[~np.isnan(s)])) if np.any(~np.isnan(s)) else np.nan
        print(
            f'{title} count-weighted significant-bin mean: '
            f'{summary["count_weighted_sig_mean"]:.3f}{unit} '
            f'(unweighted: {summary["sig_mean"]:.3f}{unit}, '
            f'{summary["sig_bins"]}/{summary["filled_bins"]} bins significant, '
            f'median |r| {med_abs_r:.3f})'
        )

        out = [fig, ax, s, headline]
        if return_p:
            out.append(p)
        if return_counts:
            out.append(counts)
        return tuple(out)
