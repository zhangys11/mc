import numpy as np
from tqdm import tqdm
from ..mcbase import McBase


class Kw_Test(McBase):

    """
    Verify the Kruskal-Wallis test statistic (H) is a X2 random variable.

    The Mann-Whitney or Wilcoxon test compares two groups while the Kruskal-Wallis test compares 3.
    Kruskal-Wallis test is a non-parametric version of one-way ANOVA. It is rank based.
    Kruskal-Wallis H: a X2 test statistic.

    Parameters
    ----------
    underlying_dist : population assumption. As KW test is non-parametric, the choice of dist doesn't matter.
        By default, we use unform.
    k : groups / classes
    n : samples per class. In this experiment, we use equal group size, i.e., n1=n2=n3=...
    """

    def __init__(self, underlying_dist='uniform', k=3, n=100, N=10000):
        super().__init__("chi2", N)
        self.underlying_dist = underlying_dist
        self.k = k
        self.n = n

    def run(self, display=True):

        nT = self.k * self.n

        Hs = []

        for _ in tqdm(range(self.N)):
            if self.underlying_dist == 'uniform':
                y1 = np.random.uniform(0, 1, self.n)
                y2 = np.random.uniform(0, 1, self.n)
                y3 = np.random.uniform(0, 1, self.n)
            else:  # 'gaussian'
                y1 = np.random.randn(self.n)  # normal
                y2 = np.random.randn(self.n)
                y3 = np.random.randn(self.n)

            yall = y1.tolist() + y2.tolist() + y3.tolist()
            sorted_id = sorted(range(len(yall)), key=lambda k: yall[k])

            R1 = np.sum(sorted_id[:self.n])
            R2 = np.sum(sorted_id[self.n:self.n + self.n])
            R3 = np.sum(sorted_id[self.n + self.n:])

            H = 12 / nT / (nT + 1) * (R1 ** 2 + R2 ** 2 + R3 ** 2) / self.n - 3 * (nT + 1)

            Hs.append(H)

        x_theory = np.linspace(np.min(Hs) - np.min(Hs), np.max(Hs) - np.min(Hs), 100)  # 差一个平移，research later
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.k-1)

        if display:
            super().hist(y=Hs,
                         title="Histogram of Kruskal-Wallis test's H statistic ($H = [{\dfrac{12}{n_{T}(n_{T}+1)}\sum_{i=1}^{k}\dfrac{R_{i}^2}{n_{i}}]-3(n_{T}+1)}$)\n. \
                         Population is " + ("U(0,1). " if self.underlying_dist == 'uniform' else "N(0,1). ") +
                               str(self.k) + " groups, " + str(self.n) + " samples per group.")
            super().plot(x=x_theory, y=theory, label='dof = ' + str(self.k - 1),
                         title='Theoretical Distribution\n$\chi^2(dof=' + str(self.k-1) + ')$')

        return
