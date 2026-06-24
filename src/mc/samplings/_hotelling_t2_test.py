import numpy as np
from tqdm import tqdm
from scipy.special import gamma
from .. import McBase


class Hotelling_T2_Test(McBase):

    """
    The Hotelling T2- distribution was proposed by H. Hotelling for testing equality of means of two normal populations.
    This functions verify the T2 statistic constructed from two multivariate Gussian follows the Hotelling's T2
    distribution.
    For k=1 the Hotelling T2- distribution reduces to the Student distribution,
    and for any k>0 it can be regarded as a multivariate generalization of the
    Student distribution
    """

    def __init__(self, n=50, k=2, N=1000):
        '''
        Parameters
        ----------
        n : samples per class.
        k : data dimensionality.
        '''
        super().__init__(None, N)
        self.k = k
        self.n = n

    def run(self, display=True):
        T2s = []

        for i in tqdm(range(self.N)):
            # Draw from a standard normal dist. The returned X is de-meaned, no need to do (X-mu) afterwards.
            X = np.random.randn(self.k, self.n)
            xbar = X.mean(axis=1, keepdims=True)
            S = np.cov(X)
            try:
                T2 = self.n * xbar.T @ np.linalg.inv(S) @ xbar
                T2s.append(T2.item())
            except np.linalg.LinAlgError:
                T2s.append(0)

        T2s = np.array(T2s)
        x_theory = np.linspace(0, np.percentile(T2s, 99.9), 100)
        theory = super().init_theory(dist='f', x_theory=x_theory,
                                      df1=self.k, df2=self.n - self.k)

        if display:
            super().hist(y=T2s, title=r"Histogram of the Hotelling's $T^2$ statistic ($T^2 = n(\overline{X}-\mu)^{T}S^{-1}(\overline{x}-\mu)$)")
            super().plot(x=x_theory, y=theory, label='$T^2(' + str(self.k) + ',' + str(self.n+self.k-1) + ')$',
                         title='Theoretical Distribution $T^2(' + str(self.k) + ',' + str(self.n+self.k-1) + ')$ \n \
                    $p(x) = \dfrac{\Gamma((n+1)/2)x^{k/2-1}(1+x/n)^{-(n+1)/2}}{\Gamma((n-k+1)/2)\Gamma(k/2)n^{k/2}}$')
