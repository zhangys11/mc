import numpy as np
from scipy import stats
from tqdm import tqdm
from .. import McBase


class Fk_Test(McBase):

    """
    Verify the Fligner-Killeen Test statistic (FK) is a X2 random variable.
    """

    def __init__(self, n=10, k=5, N=1000):
        super().__init__("chi2", N)
        self.n = n
        self.k = k

    def run(self, display=True):
        FKs = []
        nT = self.k * self.n

        for _ in tqdm(range(self.N)):
            groups = []
            for j in range(self.k):
                groups.append(np.random.normal(0, 1, self.n))
            centered = np.concatenate([np.abs(g - np.median(g)) for g in groups])
            ranks = stats.rankdata(centered)
            scores = stats.norm.ppf(0.5 + ranks / (2 * (nT + 1)))
            a_bar = scores.mean()
            idx = 0
            a_j_bars = np.zeros(self.k)
            for j in range(self.k):
                a_j_bars[j] = scores[idx:idx + self.n].mean()
                idx += self.n
            s2 = scores.var(ddof=1)
            FK = (self.n * ((a_j_bars - a_bar) ** 2).sum()) / max(s2, 1e-15)
            FKs.append(FK)

        x_theory = np.linspace(0, np.max(FKs) * 0.95, 100)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.k - 1)

        if display:
            super().hist(y=FKs,
                         title="Histogram of the Fligner-Killeen test statistic")
            super().plot(x=x_theory, y=theory, label='dof = ' + str(self.k - 1),
                         title='Theoretical Distribution\n$\chi^2(dof=' + str(self.k - 1) + ')$')
