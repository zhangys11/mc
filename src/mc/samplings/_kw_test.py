import numpy as np
from scipy import stats
from tqdm import tqdm
from .. import McBase


class Kw_Test(McBase):

    """
    Verify the Kruskal-Wallis test statistic (H) is a X2 random variable.
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
            groups = []
            for j in range(self.k):
                if self.underlying_dist == 'uniform':
                    groups.append(np.random.uniform(0, 1, self.n))
                else:
                    groups.append(np.random.randn(self.n))
            yall = np.concatenate(groups)
            ranks = stats.rankdata(yall)
            idx = 0
            rank_sums = np.zeros(self.k)
            for j in range(self.k):
                rank_sums[j] = ranks[idx:idx + self.n].sum()
                idx += self.n
            H = 12 / (nT * (nT + 1)) * (rank_sums ** 2 / self.n).sum() - 3 * (nT + 1)
            Hs.append(H)

        x_theory = np.linspace(0, np.max(Hs) * 0.95, 100)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.k - 1)

        if display:
            super().hist(y=Hs,
                         title=r"Histogram of the Kruskal-Wallis test's H statistic.\n" +
                               str(self.k) + " groups, " + str(self.n) + " samples per group.")
            super().plot(x=x_theory, y=theory, label='dof = ' + str(self.k - 1),
                         title='Theoretical Distribution\n$\chi^2(dof=' + str(self.k - 1) + ')$')
