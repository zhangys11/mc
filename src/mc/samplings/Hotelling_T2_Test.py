import numpy as np
from tqdm import tqdm
from scipy.special import gamma
from ..mcbase import McBase


class Hotelling_T2_Stat(McBase):

    """
    The Hotelling T2- distribution was proposed by H. Hotelling for testing equality of means of two normal populations.
    This functions verify the T2 statistic constructed from two multivariate Gussian follows the Hotelling's T2
    distribution.
    For k=1 the Hotelling T2- distribution reduces to the Student distribution,
    and for any k>0 it can be regarded as a multivariate generalization of the
    Student distribution

    Parameters
    ----------
    n : samples per class.
    k : data dimensionality.
    """

    def __init__(self, n=50, k=2, N=1000):
        super().__init__(None, N)
        self.k = k
        self.n = n

    def run(self, display=True):
        T2s = []

        for i in tqdm(range(self.N)):
            # Draw from a standard normal dist. The returned X is de-meaned, no need to do (X-mu) afterwards.
            X = np.random.randn(self.k, self.n)
            X_mat = np.mat(X)
            X1 = (X_mat.sum(axis=1)) / self.n
            sum_xs = 0
            for j in range(0, self.n):
                sum_xs = sum_xs + (X_mat[:, j] - X1) * (X_mat[:, j] - X1).T
            SIGMA = sum_xs / (self.n - 1)
            T2 = (self.n * X1.T) * (np.linalg.inv(SIGMA)) * X1
            T2s.append(T2[0, 0])

        x_theory = np.linspace(np.min(T2s), np.max(T2s), 100)
        theory = \
            ((gamma((self.n+1)/2))*((1+x_theory/self.n)**(-(self.n+1)/2))) / ((gamma((self.n-1)/2))*(gamma(1)*self.n))

        if display:
            super().hist(y=T2s, title="Histogram of Hotelling's $T^2$ statistic ($T^2 = n(\overline{X}-\mu)^{T}S^{-1}(\overline{x}-\mu)$)")
            super().plot(x=x_theory, y=theory, label='$T^2(' + str(self.k) + ',' + str(self.n+self.k-1) + ')$',
                         title='Theoretical Distribution $T^2(' + str(self.k) + ',' + str(self.n+self.k-1) + ')$ \n \
                    $p(x) = \dfrac{\Gamma((n+1)/2)x^{k/2-1}(1+x/n)^{-(n+1)/2}}{\Gamma((n-k+1)/2)\Gamma(k/2)n^{k/2}}$')

        return
