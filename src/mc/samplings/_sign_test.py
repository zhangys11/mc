import numpy as np
from tqdm import tqdm
from .. import McBase


class Sign_Test(McBase):

    """
    For sign test, if H0 is true (m = m0), the N- and N+ both follow b(n,1/2)
    """
    def __init__(self, underlying_dist='expon', n=100, N=1000):
        '''
        Parameters
        ----------
        underlying_dist : population assumption. As sign test is non-parametric, the choice of dist doesn't matter.
            By default, we use exponential. It's theoretical median is m = θln(2).
        n : sample size.
        '''
        super().__init__('binom', N)
        self.underlying_dist = underlying_dist
        self.n = n

    def run(self, display=True):
        poss = []
        negs = []

        for _ in tqdm(range(self.N)):
            x = np.random.exponential(scale=1, size=self.n)
            n_pos = len(np.where(x - np.log(2) > 0)[0])
            n_neg = len(np.where(x - np.log(2) < 0)[0])

            poss.append(n_pos)
            negs.append(n_neg)

        x_theory = np.linspace(0, self.n, self.n+1)
        lb = round(min(np.min(poss), np.min(negs)))
        ub = round(max(np.max(poss), np.max(negs)))
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, n=self.n, p=0.5)

        if display:
            super().hist(y=poss, title="Histogram of the sign test's N+ statistic.\n Population is expon(1). " +
                                       str(self.n) + " samples.", density=True)
            super().hist(y=negs, title="Histogram of the sign test's N- statistic.\n Population is expon(1). " +
                                       str(self.n) + " samples.", density=True)
            super().plot(x=x_theory[lb:ub], y=theory[lb:ub], label="b("+str(self.n)+',1/2)',
                         title='Theoretical Distribution\n$b(n=' + str(self.n) + ',p=1/2)$')
