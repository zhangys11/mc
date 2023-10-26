import numpy as np
from scipy.special import rel_entr
import scipy.special
import scipy.stats
from .. import McBase


class Chisq(McBase):

    """
    Construct a chisq r.v. with [n] degrees of freedom.
    The squared sum of [n] r.v.s. from standard normal distributions is a chisq statistic.
    This function will verify it via [N] MC experiments.
    The sum of n N(0,1)^2 r.v.s. is a chi-square statistic.
    """

    def __init__(self, n=10, N=10000):
        '''
        Parameters
        ----------
        n : How many r.v.s. to use
        '''
        super().__init__("chi2", N)
        self.n = n

    def run(self, display=True):

        CHISQS = []

        for _ in range(self.N):
            CHISQS.append(np.sum(np.random.randn(self.n)**2))

        ul = min(CHISQS)
        ub = max(CHISQS)+0.5
        x_theory = np.linspace(round(ul), round(ub))
        # if self.flavor == 1:
        #    theory = x_theory**(self.n/2-1)*np.exp(-x_theory/2)/(2**(self.n/2)*scipy.special.gamma(self.n/2))
        #else:
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.n)

        if display:
            super().hist(y=CHISQS, title="Frequency Histogram\nnumber of standard normal r.v.s. =" + str(self.n) + ', simulations = ' +
                                         str(self.N))
            super().plot(x=x_theory, y=theory, label=r'$\chi^2(dof=' + str(self.n) + ')$',
                         title='Theoretical Distribution\n' + r'$\chi^2(dof=' + str(self.n) + ')$')
