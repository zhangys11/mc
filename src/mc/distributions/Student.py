import numpy as np
import scipy.special
import scipy.stats
from ..mcbase import McBase


class Student(McBase):

    """
    Create a t-distribution r.v. with [k] degrees of freedom.
    """

    def __init__(self, N=10000, k=5):
        super().__init__("t", N)
        self.k = k

    def run(self, display=True):
        X = np.random.randn(self.N)
        Y = scipy.stats.chi2.rvs(df=self.k, size=self.N)
        ts = X/np.sqrt(Y/self.k)

        x_theory = np.linspace(round(min(ts)), round(max(ts)+0.5), 200)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.k)

        if display:
            super().hist(y=ts, title="Frequency Histogram\ndegree of freedom =" + str(self.k) +
                                     ', simulations = ' + str(self.N))
            super().plot(x=x_theory, y=theory, label=r'$t (dof=' + str(self.k) + ')$',
                         title='Theoretical Distribution\n' + r'$t (dof=' + str(self.k) + ')$')

        return
