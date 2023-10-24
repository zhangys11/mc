import collections
import numpy as np
from scipy.stats import binom
from ..mcbase import McBase
if __package__:
    from ..experiments import Galton_Board
else:
    from ..experiments import Galton_Board


class Chisq_Gof_Test(McBase):

    """
    Verify the chisq statistic used in Pearson's Chi-Square Goodness-of-Fit Test.
    验证皮尔逊卡方拟合优度检验的卡方分布假设

    Parameters
    ----------
    underlying_dist : what kind of population dist to use. Default we use binom, i.e., the Galton board
        'binom' / 'galton' - the population is binom
        'dice' - 6 * 1/6
    k : classes in the PMF
    """

    def __init__(self, underlying_dist='binom', k=8, sample_size=100, N=10000):
        super().__init__('chi2', N)
        self.underlying_dist = underlying_dist
        self.k = k
        self.sample_size = sample_size

    def run(self, display=True):
        # test with b(n,p)
        chisqs = []

        for i in range(self.N):  # MC试验次数
            if self.underlying_dist == 'binom' or self.underlying_dist == 'galton':
                galton_board = Galton_Board.Galton_Board(N=self.sample_size, num_layers=self.k - 1, flavor=1)
                h = galton_board.run(display=False)  # rounds, layers
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
                 Population is " + self.dist + ", sample size="+str(self.sample_size))
            super().plot(x=x_theory, y=theory, label='dof = ' + str(self.k-1),
                         title='Theoretical Distribution\n$\chi^2(dof=' + str(self.k-1) + ')$')

        return
