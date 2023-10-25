import numpy as np
from tqdm import tqdm
from ..mcbase import McBase


class Cochrane_Q_Test(McBase):
    """
    Cochrane-Q test T statistic is a X2 (CHISQ) random variable.
    Cochrane-Q is an extension of McNemar that support more than 2 samples/groups.
    H0: the percentage of "success / pass" for all groups are equal.

    Parameters
    ----------
    p : we draw from a Bernoulli population with p. p is the "success / pass" probability.
    k : groups / classes
    n : samples per class. In this experiment, all group sizes are equal, as Cochrane-Q is paired / dependent.
    """

    def __init__(self, p=0.5, k=3, n=100, N=10000):
        super().__init__('chi2', N)
        self.p = p
        self.k = k
        self.n = n

    def run(self, display=True):
        Ts = []

        for i in tqdm(range(self.N)):
            X = np.random.binomial(1, self.p, (self.n, self.k))  # return a nxK matrix of {0,1}
            T = (self.k - 1) * (self.k * np.sum(X.sum(axis=0) ** 2) -
                                np.sum(X.sum(axis=0))**2) / np.sum((self.k - X.sum(axis=1)) * X.sum(axis=1))
            Ts.append(T)

        x_theory = np.linspace(np.min(Ts), np.max(Ts), 100)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.k-1)

        if display:
            super().hist(y=Ts, title="Histogram of Cochrane-Q test's T statistic ($T = \dfrac{(k-1)[k\sum_{j=1}^{k}X_{.j}^2-(\sum_{j=1}^{k} X_{.j})^2]}{k\sum_{i=1}^{b}X_{i.}-\sum_{i=1}^{b} X_{i.}^2}$)\n. \
             Population is " + "Bernoulli(" + str(self.p) + "). " + str(self.k) + " groups, " + str(self.n) +
                                     " samples per group.")
            super().plot(x=x_theory, y=theory, label='dof = ' + str(self.k - 1),
                         title='Theoretical Distribution\n$\chi^2(dof=' + str(self.k-1) + ')$')

        return
