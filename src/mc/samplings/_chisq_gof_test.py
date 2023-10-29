import collections
import numpy as np
from scipy.stats import binom
from .. import McBase
from ..experiments import Galton_Board

class Chisq_Gof_Test(McBase):
    """
    Verify the chisq statistic used in Pearson's Chi-Square Goodness-of-Fit Test.

    Because Pearson’s chi- square GOF test is non-parametric, there is no restriction on the population distribution. 
    chisq_gof_stat() provides two MC experiment settings. 
    (1) The first is the Galton board (use the binominal population). 
    (2) The second is the dice game (use the uniform PMF). 
    In both cases, the statistic histogram from the MC experiment is very close to the theoretical χ2(k - 1) distribution.
    """

    def __init__(self, underlying_dist='binom', k=8, n=100, N=10000):
        '''
        Parameters
        ----------
        underlying_dist : what kind of population dist to use. Default we use binom, i.e., the Galton board
            'binom' / 'galton' - the population is binom
            'dice' - 6 * 1/6
        k : classes in the PMF
        n : sample size
        '''
        super().__init__('chi2', N)
        self.underlying_dist = underlying_dist
        self.k = k
        self.sample_size = n

    def run(self, display=True):
        # test with b(n,p)
        chisqs = []

        for i in range(self.N):  # MC试验次数
            if self.underlying_dist == 'binom' or self.underlying_dist == 'galton':
                galton_board = Galton_Board(N=self.sample_size, n=self.k - 1, flavor=1)
                galton_board.run(display=False)  # rounds, layers
                h = galton_board.hist
                # print('experiment', h)
                chisq = 0
                for j in range(self.k):
                    pj = binom.pmf(j, self.k - 1, 0.5)
                    npj = self.sample_size * pj  # theoretical
                    fj = h[j]
                    chisq = chisq + (fj - npj) ** 2 / npj
                chisqs.append(chisq)
            elif self.underlying_dist == 'dice':
                h = collections.Counter(np.random.randint(0, 6, self.sample_size))
                chisq = 0
                for j in range(6):
                    pj = 1.0 / 6
                    npj = self.sample_size * pj
                    fj = h[j]
                    # print(pj, npj, fj)
                    chisq = chisq + (fj - npj) ** 2 / npj
                chisqs.append(chisq)

        x_theory = np.linspace(0, np.max(chisqs), 100)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.k-1)

        if display:
            super().hist(y=chisqs, title="Histogram of the GOF test statistic ($\chi^2 = \sum_{i=1}^{k}\dfrac{(f_{j}-np_{j})^2}{np_{j}}$).\n \
                 Population is " + self.underlying_dist + ", sample size="+str(self.sample_size))
            super().plot(x=x_theory, y=theory, label='dof = ' + str(self.k-1),
                         title='Theoretical Distribution\n$\chi^2(dof=' + str(self.k-1) + ')$')