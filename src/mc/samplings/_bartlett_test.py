import numpy as np
from tqdm import tqdm
from .. import McBase


class Bartlett_Test(McBase):

    """
    Bartlett's test for homogeneity of variances. Verify the statistic is X2.
    """

    def __init__(self, k=5, n=10, N=1000):
        super().__init__('chi2', N)
        self.k = k
        self.n = n

    def run(self, display=True):
        BTs = []
        N_total = self.k * self.n

        for _ in tqdm(range(self.N)):
            X = np.random.randn(self.k, self.n)
            Si_2 = np.var(X, axis=1, ddof=1)
            S_p2 = sum((self.n - 1) * Si_2) / (N_total - self.k)
            num = (N_total - self.k) * np.log(S_p2) - (self.n - 1) * np.log(Si_2).sum()
            C = 1 + (1 / (3 * (self.k - 1))) * (self.k / (self.n - 1) - 1 / (N_total - self.k))
            BTs.append(num / C)

        x_theory = np.linspace(0, np.max(BTs) * 0.95, 100)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.k - 1)

        if display:
            super().hist(y=BTs, title="Histogram of Bartlett's test statistic")
            super().plot(x=x_theory, y=theory, label='dof = ' + str(self.k - 1),
                         title='Theoretical Distribution\n$\chi^2(dof=' + str(self.k - 1) + ')$')
