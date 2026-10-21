"""Mardia's multivariate skewness and kurtosis tests of multivariate normality.

Mardia (1970) generalises the univariate skewness and kurtosis to ``p``-variate
data.  For ``X`` with mean ``mu`` and covariance ``Sigma`` define the squared
Mahalanobis distance ``D^2 = (X - mu)' Sigma^{-1} (X - mu)``.  The population
measures are

    beta_{1,p} = E[ ((X - mu)' Sigma^{-1} (Y - mu))^3 ]      (skewness)
    beta_{2,p} = E[ (D^2)^2 ]                                 (kurtosis)

with ``X``, ``Y`` independent copies.  For a multivariate normal population
``beta_{1,p} = 0`` and ``beta_{2,p} = p (p + 2)``.  For the multivariate t
distribution with ``nu > 4`` degrees of freedom and its true covariance matrix,
``beta_{2,p} = p (p + 2) (nu - 2) / (nu - 4)``.

The sample versions use the covariance matrix with divisor ``n``,

    S = (1/n) sum_i (X_i - Xbar)(X_i - Xbar)',

and the Mahalanobis products ``d_ij = (X_i - Xbar)' S^{-1} (X_j - Xbar)``:

    b_{1,p} = (1/n^2) sum_i sum_j d_ij^3,
    b_{2,p} = (1/n) sum_i d_ii^2.

The asymptotic normal-theory references are

    n b_{1,p} / 6  ->  chi2( p (p + 1) (p + 2) / 6 ),
    Z_K = (b_{2,p} - p (p + 2)) / sqrt(8 p (p + 2) / n)  ->  N(0, 1).

Both references are asymptotic.  In finite samples the kurtosis statistic is
biased downwards: for multivariate normal data

    E[b_{2,p}] = p (p + 2) (n - 1) / (n + 1),

so the nominal normal reference is not centred.  The class reports this
finite-sample expectation; the accompanying study (`mardia/`) quantifies the
resulting miscalibration by Monte Carlo.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from .. import BARPLOT_KWARGS, McBase


class Mardia_Test(McBase):

    """
    Verify Mardia's multivariate skewness and kurtosis statistics against their
    asymptotic normal-theory null distributions.

    Parameters
    ----------
    n : int
        Sample size.  Must exceed p + 1 for a non-singular sample covariance
        matrix (and for the finite-sample kurtosis result below).
    p : int
        Number of variables, p >= 1.
    N : int
        Number of MC experiments to run.
    parent : str
        Marginal population: 'normal' (default), 't', 'exp', 'chi2' or
        'lognorm'.  For 't' the multivariate t is drawn with its true
        covariance standardized to the identity; the other non-normal choices
        use independent standardized margins with identity covariance, so the
        population covariance is the identity in every case.
    parent_args : dict or None
        Extra parameters: {'df': ...} for 't' (default 10) and 'chi2'
        (default 1), {'sigma': ...} for 'lognorm' (default 0.5).
    seed : int or None
        Optional seed, for reproducible simulations.
    """

    PARENT_NAMES = ("normal", "t", "exp", "chi2", "lognorm")

    def __init__(self, n=50, p=2, N=1000, parent="normal",
                 parent_args=None, seed=None):
        super().__init__("chi2", N)

        # A private generator keeps the simulation reproducible without touching
        # the global numpy RNG that McBase seeds.
        self.seed = seed
        self.n = int(n)
        self.p = int(p)
        if self.p < 1:
            raise ValueError("p must be at least 1")
        if self.n <= self.p + 1:
            raise ValueError(
                "n must exceed p + 1 so that the sample covariance matrix is "
                "non-singular and the finite-sample kurtosis result applies")
        if parent not in self.PARENT_NAMES:
            raise ValueError("parent must be one of {}".format(self.PARENT_NAMES))
        self.parent = parent
        self.parent_args = dict(parent_args or {})

        self.stats = None      # filled by simulate()
        self.summary_ = None   # filled by run()/summary()

    # ------------------------------------------------------------------ #
    # reference distributions / finite-sample moments
    # ------------------------------------------------------------------ #
    @property
    def dof_skew(self):
        """Degrees of freedom p (p + 1) (p + 2) / 6 of the skewness reference."""
        return self.p * (self.p + 1) * (self.p + 2) // 6

    @property
    def expected_kurtosis(self):
        """Normal-theory value p (p + 2) of the population kurtosis."""
        return self.p * (self.p + 2)

    @property
    def expected_kurtosis_finite(self):
        """Exact finite-sample E[b2] = p (p + 2) (n - 1) / (n + 1) under normality."""
        return self.p * (self.p + 2) * (self.n - 1.0) / (self.n + 1.0)

    # ------------------------------------------------------------------ #
    # data generation
    # ------------------------------------------------------------------ #
    def _draw(self, rng):
        """Draw one (n, p) sample from the chosen parent with identity covariance."""
        shape = (self.n, self.p)
        args = self.parent_args

        if self.parent == "normal":
            return rng.standard_normal(shape)
        if self.parent == "t":
            df = float(args.get("df", 10.0))
            if df <= 4:
                raise ValueError("t parent needs df > 4 for a finite kurtosis")
            # Z / sqrt(W/df) has covariance df/(df-2) I; rescale to identity.
            return rng.standard_normal(shape) * np.sqrt(
                (df - 2.0) / rng.chisquare(df, (self.n, 1)))
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

    # ------------------------------------------------------------------ #
    # statistics
    # ------------------------------------------------------------------ #
    @staticmethod
    def statistics(X):
        """
        Compute Mardia's statistics for one data set.

        Parameters
        ----------
        X : array_like, shape (n, p)
            Observations; the sample covariance matrix uses divisor n.

        Returns
        -------
        dict with keys 'b1', 'b2' (sample skewness / kurtosis), 'skew'
        (n b1 / 6) and 'kurt' (the standardized kurtosis Z_K).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a one- or two-dimensional array")
        n, p = X.shape
        if n <= p + 1:
            raise ValueError("n must exceed p + 1")

        centered = X - X.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / n
        inv_cov = np.linalg.inv(cov)
        # D[i, j] = (X_i - Xbar)' S^{-1} (X_j - Xbar)
        d = centered @ inv_cov @ centered.T

        b1 = float(np.sum(d ** 3) / n ** 2)
        b2 = float(np.sum(np.diag(d) ** 2) / n)

        return {
            "b1": b1,
            "b2": b2,
            "skew": float(n * b1 / 6.0),
            "kurt": float((b2 - p * (p + 2)) / np.sqrt(8.0 * p * (p + 2) / n)),
        }

    def test(self, X):
        """
        Apply Mardia's tests to one data set and return statistics with
        asymptotic p-values.

        The p-values rely on the normal-theory references and are the ones the
        test is normally used with; they inherit the finite-sample bias
        documented in the class docstring.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        result = self.statistics(X)
        p = X.shape[1]
        dof = p * (p + 1) * (p + 2) // 6
        result["p_skew"] = float(stats.chi2.sf(result["skew"], dof))
        result["p_kurt"] = float(2.0 * stats.norm.sf(abs(result["kurt"])))
        result["p_kurt_upper"] = float(stats.norm.sf(result["kurt"]))
        return result

    # ------------------------------------------------------------------ #
    # simulation
    # ------------------------------------------------------------------ #
    def simulate(self, progress=True):
        """
        Run the Monte Carlo experiment.

        Returns
        -------
        dict with arrays 'b1', 'b2', 'skew', 'kurt' of length N.
        """
        keys = ("b1", "b2", "skew", "kurt")
        out = {key: np.empty(self.N) for key in keys}
        rng = np.random.default_rng(self.seed)

        for i in tqdm(range(self.N), disable=not progress):
            row = self.statistics(self._draw(rng))
            for key in keys:
                out[key][i] = row[key]

        self.stats = out
        return out

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def summary(self, alpha=0.05, progress=True):
        """
        Summarize the simulated null distributions against the asymptotic
        references and report the finite-sample kurtosis bias.

        Returns
        -------
        dict with the sample moments of the two statistics, the KS distance to
        chi2(dof_skew) and N(0, 1), the empirical rejection rates at level
        ``alpha`` and the exact finite-sample expectation of b2.
        """
        if self.stats is None:
            self.simulate(progress=progress)

        skew = self.stats["skew"]
        kurt = self.stats["kurt"]
        b2 = self.stats["b2"]
        dof = self.dof_skew

        out = {
            "p": self.p,
            "n": self.n,
            "N": self.N,
            "parent": self.parent,
            "parent_args": dict(self.parent_args),
            "dof_skew": dof,
            "expected_kurtosis": self.expected_kurtosis,
            "expected_kurtosis_finite": self.expected_kurtosis_finite,
            "mean_skew": float(np.mean(skew)),
            "var_skew": float(np.var(skew, ddof=1)),
            "mean_chi2": float(dof),
            "var_chi2": float(2 * dof),
            "mean_b2": float(np.mean(b2)),
            "mean_kurt": float(np.mean(kurt)),
            "sd_kurt": float(np.std(kurt, ddof=1)),
            "ks_skew": float(stats.kstest(skew, stats.chi2.cdf, args=(dof,)).statistic),
            "ks_kurt": float(stats.kstest(kurt, stats.norm.cdf).statistic),
            "alpha": alpha,
        }

        skew_critical = stats.chi2.ppf(1.0 - alpha, dof)
        out["critical_skew"] = float(skew_critical)
        out["type1_skew"] = float(np.mean(skew > skew_critical))

        kurt_critical = stats.norm.ppf(1.0 - alpha / 2.0)
        out["critical_kurt"] = float(kurt_critical)
        out["type1_kurt"] = float(np.mean(np.abs(kurt) > kurt_critical))
        out["type1_kurt_upper"] = float(
            np.mean(kurt > stats.norm.ppf(1.0 - alpha)))

        return out

    def run(self, display=True):
        """
        Simulate both null distributions and plot them against their
        asymptotic references.  Diagnostic results are stored in
        ``self.summary_``.
        """
        self.simulate()
        skew = self.stats["skew"]
        kurt = self.stats["kurt"]
        self.summary_ = self.summary()

        if display:
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))

            upper_skew = float(np.quantile(skew, 0.999))
            grid_skew = np.linspace(0.0, upper_skew, 200)
            axes[0].hist(skew, bins=80, density=True, label="MC null",
                         **BARPLOT_KWARGS)
            axes[0].plot(grid_skew, stats.chi2.pdf(grid_skew, self.dof_skew),
                         color="black", lw=1.6,
                         label="chi2(dof = {})".format(self.dof_skew))
            axes[0].set(title="Mardia skewness statistic\n"
                              "p = {}, n = {}".format(self.p, self.n),
                        xlabel="n b1 / 6", ylabel="density")
            axes[0].legend(fontsize=8)
            axes[0].grid(alpha=0.3)

            upper_kurt = float(np.quantile(np.abs(kurt), 0.999))
            grid_kurt = np.linspace(-1.05 * upper_kurt, 1.05 * upper_kurt, 300)
            axes[1].hist(kurt, bins=80, density=True, label="MC null",
                         **BARPLOT_KWARGS)
            axes[1].plot(grid_kurt, stats.norm.pdf(grid_kurt),
                         color="black", lw=1.6, label="N(0, 1)")
            axes[1].set(title="Mardia kurtosis statistic\n"
                              "p = {}, n = {}".format(self.p, self.n),
                        xlabel="Z_K", ylabel="density")
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3)

            fig.tight_layout()
            plt.show()

        return self.summary_
