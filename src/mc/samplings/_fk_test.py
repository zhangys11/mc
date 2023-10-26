import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
from .. import McBase


class Fk_Test(McBase):

    """
    Verify the Fligner-Killeen Test statistic (FK) is a X2 random variable.
    The Fligner-Killeen test is a non-parametric test for homogeneity of group variances based on ranks.
    """

    def __init__(self, n=10, k=5, N=1000):
        '''
        Parameters
        ----------
        n : samples per class. In this experiment, all group sizes are equal.
        k : groups / classes
        '''
        super().__init__("chi2", N)
        self.n = n
        self.k = k

    def run(self, display=True):
        FKs = []
        for _ in tqdm(range(self.N)):
            X = np.random.randint(0, 100, [self.k, self.n])
            X_normal = preprocessing.scale(X)
            a_j_bar = (X_normal.sum(axis=1)) / self.n
            a_bar = X_normal.sum() / (self.n * self.k)
            total = 0
            for j in range(0, self.k):
                total = total + self.k * (a_j_bar[j] - a_bar) ** 2
            FK = total / X_normal.var()
            FKs.append(FK)

        x_theory = np.linspace(np.min(FKs), np.max(FKs), 100)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.k-1)
        if display:
            super().hist(y=FKs, title="Histogram of the Fligner-Killeen test statistic ($FK = \dfrac{\sum_{j=1}^{k}n_{j}(\overline{a_{j}}-\overline{a})^2}{s^2}$)")
            super().plot(x=x_theory, y=theory, label='dof = ' + str(self.k - 1),
                         title='Theoretical Distribution\n$\chi^2$(dof=' + str(self.k-1) + ')')
