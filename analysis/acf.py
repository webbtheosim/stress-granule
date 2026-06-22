"""
Stress autocorrelation function (ACF) for Green-Kubo viscosity.

Support module imported by ``viscosity.py``. Computes the time autocorrelation of the
three off-diagonal stress-tensor components (Pxy, Pxz, Pyz) of the condensate,
averaged per Green-Kubo convention, with optional moving-window block averaging
and bootstrap resampling over contiguous trajectory segments.

Key inputs:
    p_xyz : array of shape (n_frames, 3) holding the off-diagonal stress
            components in consistent (Pa) units, sampled at a fixed cadence.

Key outputs:
    lag times (in sample units) and one or more autocorrelation curves
    (bootstrapped or per-segment), consumed downstream by ``VSC.VSC.get_visco``
    to integrate the Green-Kubo relation.

Not runnable as a script; ``ACF`` is instantiated by the viscosity pipeline.
"""

import numpy as np

class acf:
    """Stress autocorrelation estimator (off-diagonal Green-Kubo kernels)."""

    def __init__(self):
        """Create a stateless ACF helper (all data passed per method call)."""
        pass

    @staticmethod
    def segment_slices(n_points, n_segments):
        """Return ``n_segments`` contiguous equal-width ``(start, stop)`` index
        slices spanning ``n_points``; empty list if the data is too short."""
        if n_points <= 0 or n_segments <= 0:
            return []
        width = int(n_points / n_segments)
        if width <= 1:
            return []
        return [(width * i, width * (i + 1)) for i in range(n_segments)]

    def multi_kernel(self, x, t=1):
        """Lag-``t`` product kernel for one stress component (``t`` in {0, 1}).

        Returns ``x[:-t] * x[t:]`` for ``t == 1`` (lag-1 product) or ``x * x``
        for ``t == 0`` (zero-lag variance term). Used only for the smallest lags;
        larger lags go through :meth:`multi_kernel_block`.

        Raises:
            ValueError: if ``t`` is anything other than 0 or 1.
        """
        if t == 1:
            result = np.multiply(x[:-t], x[t:])
        elif t == 0:
            result = np.multiply(x[:], x[:])
        else:
            raise ValueError(f"multi_kernel supports only t in {{0, 1}}; got t={t}")
        return result

    def multi_kernel_block(self, x, t=1):
        """Block-averaged lag-``t`` product kernel for one stress component.

        Splits ``x`` into consecutive blocks of length ``t``, averages each
        block, then returns the product of adjacent block means. Block averaging
        before multiplying suppresses high-frequency noise relative to taking the
        raw product of samples ``t`` apart.
        """
        n_ind = len(x) // t
        indices = np.arange(n_ind) * t + np.arange(t)[:, None]
        subarrays = x[indices]
        mean = np.mean(subarrays, axis=0)
        result = np.multiply(mean[:-1], mean[1:])
        return result


    def get_boot_data(self, p_xyz, segment, bootstrap, seed=0):
        """Build bootstrapped stress autocorrelation functions.

        Partitions ``p_xyz`` into ``segment`` contiguous equal-length windows,
        computes one ACF per window (off-diagonals averaged per Green-Kubo
        convention), then resamples those per-window ACFs with replacement
        ``bootstrap`` times and averages each draw.

        Args:
            p_xyz (array-like): Off-diagonal stress tensor, shape (n_frames, 3).
            segment (int): Number of contiguous segments to partition the data.
            bootstrap (int): Number of bootstrap resamples to draw.
            seed (int): RNG seed for reproducible resampling.

        Returns:
            tuple: ``(dt, acfs_bootstrap)`` where ``dt`` is the array of lag
            indices and ``acfs_bootstrap`` is the bootstrapped ACF ensemble; both
            are empty when the data is too short to segment.
        """
        slices = self.segment_slices(len(p_xyz), segment)
        if not slices:
            return np.array([], dtype=int), np.array([])

        w = slices[0][1] - slices[0][0]  # width of each non-overlapping segment
        dt = np.arange(0, int(w / 2), 1)

        acfs = []
        for start, stop in slices:
            xyz = p_xyz[start:stop]
            acf = []
            for j in dt:
                if j <= 1:
                    acf_temp = (
                        self.multi_kernel(xyz[..., 0], j).mean()
                        + self.multi_kernel(xyz[..., 1], j).mean()
                        + self.multi_kernel(xyz[..., 2], j).mean()
                    ) / 3.0
                else:
                    acf_temp = (
                        self.multi_kernel_block(xyz[..., 0], j).mean()
                        + self.multi_kernel_block(xyz[..., 1], j).mean()
                        + self.multi_kernel_block(xyz[..., 2], j).mean()
                    ) / 3.0
                acf.append(acf_temp)
            acfs.append(np.array(acf))

        acfs = np.array(acfs)

        rng = np.random.RandomState(seed)
        acfs_bootstrap = []
        for _ in range(bootstrap):
            boot_idx = rng.randint(0, len(acfs), size=len(acfs))
            acfs_bootstrap.append(acfs[boot_idx].mean(axis=0))
        acfs_bootstrap = np.array(acfs_bootstrap)

        return dt, acfs_bootstrap

    def segment_acfs(self, p_xyz, segment):
        """
        Partition the pressure tensor into contiguous equal-length segments and
        return one autocorrelation function per segment without bootstrapping.
        """

        p_xyz = np.asarray(p_xyz)
        slices = self.segment_slices(len(p_xyz), segment)
        if not slices:
            return np.array([], dtype=int), np.array([]), []

        acfs = []
        out_slices = []
        dt = None
        for start, stop in slices:
            xyz = p_xyz[start:stop]
            if xyz.ndim != 2 or xyz.shape[0] < 4 or xyz.shape[1] != 3:
                continue

            local_dt = np.arange(0, int(xyz.shape[0] / 2), 1)
            if local_dt.size == 0:
                continue

            acf = []
            for lag in local_dt:
                if lag <= 1:
                    acf_temp = (
                        self.multi_kernel(xyz[..., 0], lag).mean()
                        + self.multi_kernel(xyz[..., 1], lag).mean()
                        + self.multi_kernel(xyz[..., 2], lag).mean()
                    ) / 3.0
                else:
                    acf_temp = (
                        self.multi_kernel_block(xyz[..., 0], lag).mean()
                        + self.multi_kernel_block(xyz[..., 1], lag).mean()
                        + self.multi_kernel_block(xyz[..., 2], lag).mean()
                    ) / 3.0
                acf.append(acf_temp)

            acf = np.asarray(acf, dtype=float)
            if acf.size == 0:
                continue
            if dt is None:
                dt = local_dt
            elif dt.shape != local_dt.shape:
                continue

            acfs.append(acf)
            out_slices.append((start, stop))

        if dt is None or not acfs:
            return np.array([], dtype=int), np.array([]), []
        return dt, np.asarray(acfs, dtype=float), out_slices
