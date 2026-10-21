"""Box's M test for the homogeneity of covariance matrices across k groups.

Box's M (Box, 1949) generalises Bartlett's test of equal variances to the
multivariate case.  For ``k`` groups of ``p``-variate observations with sample
covariance matrices ``S_j`` and pooled covariance matrix ``S_p``, the null
hypothesis is

    H0: Sigma_1 = Sigma_2 = ... = Sigma_k,

and the statistic is

    M = (N - k) ln|S_p| - sum_j (n_j - 1) ln|S_j|,

where ``N = sum_j n_j``.  Under multivariate normality a Bartlett-type
correction ``c`` gives

    (1 - c) M  ~  chi2(nu),   nu = (k - 1) p (p + 1) / 2,

approximately, with

    c = (2p^2 + 3p - 1) / (6 (p + 1) (k - 1))
        * ( sum_j 1/(n_j - 1) - 1/(N - k) ).

The normal-theory reference is the weak point of the test.  Sample covariance
matrices fluctuate with the fourth moments of the population, not only with
its covariance matrix, so for non-normal populations the null law of ``M`` is
in general a *weighted* sum of chi-square variables and the nominal
chi-square reference is badly calibrated (the test is severely liberal).  The
``parent`` argument of :class:`Box_M_Test` draws from non-normal populations
with exactly the same covariance matrix, so the class can demonstrate both the
normal-theory adequacy and its breakdown.
"""

import numpy as np
from scipy import stats
from tqdm import tqdm

from .. import McBase


class Box_M_Test(McBase):

    """
    Verify that the Bartlett-corrected Box's M statistic is a chi-square
    random variable.  M tests H0: Sigma_1 = ... = Sigma_k.

    Parameters
    ----------
    k : int
        Number of groups (populations), k >= 2.
    p : int
        Number of variables in each group, p >= 1.
    n : int or sequence of int
        Sample size per group.  A scalar means equal group sizes; a sequence
        of length k allows unequal (unbalanced) groups.
    N : int
        Number of MC experiments to run.
    parent : str
        Marginal population of the simulated data: 'normal' (default), 't',
        'exp', 'chi2' or 'lognorm'.  All groups share the same covariance
        matrix for every choice, so H0 is true by construction; the
        non-normal choices are standardized to mean 0 and variance 1.
        Box's M is invariant under a common nonsingular linear transformation,
        so the common covariance matrix may be taken to be the identity.
    parent_args : dict or None
        Extra parameters for the marginal family: {'df': ...} for 't' and
        'chi2', {'sigma': ...} for 'lognorm'.  Defaults: df = 8 for 't',
        df = 1 for 'chi2', sigma = 0.5 for 'lognorm'.
    seed : int or None
        Optional seed, for reproducible simulations.
    """

    PARENT_NAMES = ("normal", "t", "exp", "chi2", "lognorm")

    def __init__(self, k=3, p=3, n=20, N=1000, parent="normal",
                 parent_args=None, seed=None):
        super().__init__("chi2", N)

        # A private generator keeps the simulation reproducible without touching
        # the global numpy RNG that McBase seeds.
        self.seed = seed
        self.k = int(k)
        self.p = int(p)
        if self.k < 2:
            raise ValueError("k must be at least 2")
        if self.p < 1:
            raise ValueError("p must be at least 1")

        sizes = np.atleast_1d(n).astype(int)
        if sizes.size == 1:
            sizes = np.repeat(sizes, self.k)
        if sizes.size != self.k:
            raise ValueError("n must be a scalar or contain one entry per group")
        if np.any(sizes - 1 < self.p):
            raise ValueError(
                "each group needs at least p + 1 observations for a "
                "non-singular sample covariance matrix")
        if sizes.sum() - self.k <= 0:
            raise ValueError("not enough observations")

        self.group_sizes = tuple(int(x) for x in sizes)

        if parent not in self.PARENT_NAMES:
            raise ValueError("parent must be one of {}".format(self.PARENT_NAMES))
        self.parent = parent
        self.parent_args = dict(parent_args or {})

        self.stats = None      # filled by simulate()
        self.summary_ = None   # filled by run()/summary()

    # ------------------------------------------------------------------ #
    # statistic
    # ------------------------------------------------------------------ #
    @property
    def dof(self):
        """Degrees of freedom nu = (k - 1) p (p + 1) / 2 of the chi-square reference."""
        return (self.k - 1) * self.p * (self.p + 1) // 2

    @property
    def correction(self):
        """Bartlett-type finite-sample correction factor c; the test uses (1 - c) M."""
        sizes = np.asarray(self.group_sizes, dtype=float)
        n_total = sizes.sum()
        n_minus_k = n_total - self.k
        factor = (2 * self.p ** 2 + 3 * self.p - 1) / \
                 (6 * (self.p + 1) * (self.k - 1))
        c = factor * (np.sum(1.0 / (sizes - 1.0)) - 1.0 / n_minus_k)
        return float(c)

    @staticmethod
    def _logdet(matrix):
        sign, value = np.linalg.slogdet(matrix)
        if sign <= 0:
            raise np.linalg.LinAlgError("sample covariance matrix is singular")
        return value

    def statistic(self, groups):
        """
        Compute Box's M, the correction c and the corrected statistic (1 - c) M.

        Parameters
        ----------
        groups : list of array_like
            One (n_j, p) array per group.  All arrays must have the same p.
            A one-dimensional array is treated as a single variable (p = 1).

        Returns
        -------
        dict with keys 'M', 'c', 'statistic' (= (1 - c) M).
        """
        k = len(groups)
        if k < 2:
            raise ValueError("Box's M needs at least two groups")

        arrays = [np.asarray(g, dtype=float) for g in groups]
        arrays = [g.reshape(-1, 1) if g.ndim == 1 else g for g in arrays]
        if any(g.ndim != 2 for g in arrays):
            raise ValueError("each group must be a one- or two-dimensional array")
        p = arrays[0].shape[1]
        if any(g.shape[1] != p for g in arrays):
            raise ValueError("all groups must have the same number of variables")
        sizes = np.array([g.shape[0] for g in arrays], dtype=float)
        if np.any(sizes - 1 < p):
            raise ValueError(
                "each group needs at least p + 1 observations for a "
                "non-singular sample covariance matrix")
        n_total = sizes.sum()
        n_minus_k = n_total - k

        covs = [np.atleast_2d(np.cov(g, rowvar=False, ddof=1)) for g in arrays]
        pooled = sum((sizes[j] - 1.0) * covs[j] for j in range(k)) / n_minus_k

        m_stat = n_minus_k * self._logdet(pooled) - \
            sum((sizes[j] - 1.0) * self._logdet(covs[j]) for j in range(k))

        factor = (2 * p ** 2 + 3 * p - 1) / (6 * (p + 1) * (k - 1))
        c = factor * (np.sum(1.0 / (sizes - 1.0)) - 1.0 / n_minus_k)

        return {"M": float(m_stat), "c": float(c),
                "statistic": float((1.0 - c) * m_stat)}

    # ------------------------------------------------------------------ #
    # simulation
    # ------------------------------------------------------------------ #
    def _draw_group(self, rng, size):
        """Draw one group of iid standardized p-variate observations."""
        shape = (size, self.p)
        args = self.parent_args

        if self.parent == "normal":
            return rng.standard_normal(shape)
        if self.parent == "t":
            df = float(args.get("df", 8.0))
            if df <= 2:
                raise ValueError("t parent needs df > 2")
            return rng.standard_t(df, shape) / np.sqrt(df / (df - 2.0))
        if self.parent == "exp":
            return rng.exponential(1.0, shape) - 1.0
        if self.parent == "chi2":
            df = float(args.get("df", 1.0))
            return (rng.chisquare(df, shape) - df) / np.sqrt(2.0 * df)
        if self.parent == "lognorm":
            sigma = float(args.get("sigma", 0.5))
            mean = np.exp(sigma ** 2 / 2.0)
            sd = np.sqrt((np.exp(sigma ** 2) - 1.0) * np.exp(sigma ** 2))
            return (rng.lognormal(0.0, sigma, shape) - mean) / sd

        raise ValueError("unknown parent {!r}".format(self.parent))

    def simulate(self, progress=True):
        """
        Run the Monte Carlo experiment.

        Returns
        -------
        dict with the raw statistics 'M' and the corrected statistics
        'statistic' (arrays of length N).
        """
        rng = np.random.default_rng(self.seed)
        raw = np.empty(self.N)
        for i in tqdm(range(self.N), disable=not progress):
            groups = [self._draw_group(rng, size) for size in self.group_sizes]
            raw[i] = self.statistic(groups)["M"]

        self.stats = {"M": raw, "statistic": (1.0 - self.correction) * raw}
        return self.stats

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def summary(self, alpha=0.05, progress=True):
        """
        Summarize the simulated null distribution against the chi-square reference.

        Returns
        -------
        dict with the sample mean/variance, the Kolmogorov-Smirnov distance to
        chi2(dof), the empirical type I error at level ``alpha`` and a
        two-moment (Satterthwaite) scaled-chi-square approximation.  The latter
        is a diagnostic, not the exact null law: for non-normal populations the
        exact limit is a weighted sum of chi-squares (see the ``box_m`` study).
        """
        if self.stats is None:
            self.simulate(progress=progress)

        corrected = self.stats["statistic"]
        raw = self.stats["M"]
        nu = self.dof

        mean = float(np.mean(corrected))
        var = float(np.var(corrected, ddof=1))
        ks = float(stats.kstest(corrected, stats.chi2.cdf, args=(nu,)).statistic)
        critical = stats.chi2.ppf(1.0 - alpha, nu)
        type1 = float(np.mean(corrected > critical))

        out = {
            "k": self.k,
            "p": self.p,
            "n": self.group_sizes,
            "N": self.N,
            "parent": self.parent,
            "parent_args": dict(self.parent_args),
            "dof": nu,
            "correction": self.correction,
            "mean": mean,
            "var": var,
            "mean_chi2": float(nu),
            "var_chi2": float(2 * nu),
            "ks_chi2": ks,
            "alpha": alpha,
            "critical": float(critical),
            "type1": type1,
            "mean_M": float(np.mean(raw)),
            "type1_uncorrected": float(np.mean(raw > critical)),
        }

        # Two-moment scaled chi-square a * chi2(b): the "merely scaled"
        # approximation, which is what a single Bartlett factor can capture.
        if var > 0.0 and mean > 0.0:
            b = 2.0 * mean ** 2 / var
            a = var / (2.0 * mean)
            out["fit_scale"] = float(a)
            out["fit_dof"] = float(b)
            out["ks_scaled_chi2"] = float(stats.kstest(
                corrected, lambda x: stats.chi2.cdf(x / a, b)).statistic)

        return out

    def run(self, display=True):
        """
        Simulate the null distribution and compare it with chi2(dof).

        The corrected statistics are stored in ``self.stats`` and the
        diagnostics of :meth:`summary` in ``self.summary_``.
        """
        self.simulate()
        corrected = self.stats["statistic"]
        self.summary_ = self.summary()

        upper = float(np.quantile(corrected, 0.999))
        x_theory = np.linspace(0.0, upper, 200)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.dof)

        title = ("Histogram of Box's M statistic\n"
                 "k = {}, p = {}, n = {}, corrected".format(
                     self.k, self.p, self.group_sizes))

        if display:
            super().hist(y=corrected, title=title)
            super().plot(x=x_theory, y=theory,
                         label="dof = {}".format(self.dof),
                         title="Theoretical Distribution\n"
                               "$\\chi^2(\\mathrm{{dof}}={})$".format(self.dof))

        return self.summary_
