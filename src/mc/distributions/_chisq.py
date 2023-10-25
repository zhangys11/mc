import numpy as np
from scipy.special import rel_entr
import scipy.special
import scipy.stats
from ..mcbase import McBase


class Chisq(McBase):

    """
    Construct a chisq r.v. with [k] degrees of freedom.
    The squared sum of [k] r.v.s. from standard normal distributions is a chisq statistic.
    This function will verify it via [N] MC experiments.
    [k]个 N(0,1)^2 r.v.s. 的和为一个卡方分布的统计量

    Parameters
    ----------
    k : How many r.v.s. to use
    """

    def __init__(self, k=10, N=10000, flavor=1):
        super().__init__("chi2", N)
        self.k = k
        self.flavor = flavor

    def run(self, display=True):

        CHISQS = []

        for _ in range(self.N):
            CHISQS.append(np.sum(np.random.randn(self.k)**2))

        ul = min(CHISQS)
        ub = max(CHISQS)+0.5
        x_theory = np.linspace(round(ul), round(ub))
        if self.flavor == 1:
            theory = x_theory**(self.k/2-1)*np.exp(-x_theory/2)/(2**(self.k/2)*scipy.special.gamma(self.k/2))
        else:
            theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.k)

        if display:
            super().hist(y=CHISQS, title="Frequency Histogram\ndegree of freedom =" + str(self.k) + ', simulations = ' +
                                         str(self.N))
            super().plot(x=x_theory, y=theory, label=r'$\chi^2(dof=' + str(self.k) + ')$',
                         title='Theoretical Distribution\n' + r'$\chi^2(dof=' + str(self.k) + ')$')

        return
