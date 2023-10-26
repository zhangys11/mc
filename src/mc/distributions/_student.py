import numpy as np
import scipy.special
import scipy.stats
from .. import McBase


class Student(McBase):

    """
    Create a t-distribution r.v. with [n] degrees of freedom.
    """

    def __init__(self, N=10000, n=5):
        super().__init__("t", N)
        self.n = n

    def run(self, display=True):
        X = np.random.randn(self.N)
        Y = scipy.stats.chi2.rvs(df=self.n, size=self.N)
        ts = X/np.sqrt(Y/self.n)

        x_theory = np.linspace(round(min(ts)), round(max(ts)+0.5), 200)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.n)

        if display:
            super().hist(y=ts, title="Frequency Histogram\ndof =" + str(self.n) +
                                     ', simulations = ' + str(self.N))
            super().plot(x=x_theory, y=theory, label=r'$t (dof=' + str(self.n) + ')$',
                         title='Theoretical Distribution\n' + r'$t (dof=' + str(self.n) + ')$')
