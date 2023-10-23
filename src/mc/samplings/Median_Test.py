import numpy as np
from scipy.stats import chi2
from tqdm import tqdm
from ..mcbase import McBase


class Median_Stat(McBase):

    """
    This test is performed by analyzing multiple sets of independent samples.
    Examine whether there is a significant difference in the median of the population from which they come.

    Parameters
    ----------
    n : samples per class. In this experiment, all group sizes are equal.
    k : groups / classes
    """

    def __init__(self, n=1000, k=5, N=10000):
        super().__init__("chi2", N)
        self.n = n
        self.k = k

    def run(self, display=True):

        MTs = []

        for _ in tqdm(range(self.N)):
            X = np.random.randint(0, 100, [self.k, self.n])
            Os = []
            for j in range(0, self.k):
                x_median = np.median(X[j])
                O_1i = 0
                for y in range(0, self.n):
                    if X[j][y] > x_median:
                        O_1i += 1
                Os.append(O_1i)

            X_median = np.median(X)
            a = 0
            for j in range(0, self.k):
                for y in range(0, self.n):
                    if X[j][y] > X_median:
                        a += 1
            accu = 0
            for x in range(0, self.k):
                accu += ((Os[x] - (self.n * a) / (self.n * self.k)) ** 2) / self.n
            MT = ((self.n * self.k) ** 2 / (a * ((self.n * self.k) - a))) * accu
            MTs.append(MT)

        x_theory = np.linspace(chi2.ppf(0.0001, df=self.k-1), chi2.ppf(0.9999, df=self.k-1), 100)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.k-1)

        if display:
            super().hist(y=MTs, title="Histogram of Median Test $MT$ statistic ($MT = \dfrac{N^2}{ab}\sum_{i=1}^{k}\dfrac{(O_{1i}-n_{i}a/N)^2}{n_{i}}$)")
            super().plot(x=x_theory, y=theory, label='dof = ' + str(self.k - 1),
                         title='Theoretical Distribution\n$\chi^2$(dof=' + str(self.k-1) + ')')

        return
