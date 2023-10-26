import collections
import scipy.special
import scipy.stats
from .. import McBase


class Poisson(McBase):

    """
    The poisson distribution is a limit distribution of binom b(n,p) when n is large and p is small.
    This class will approximate poisson by binom.

    The Poisson distribution has the following PMF:P (X = k) = λk e−λ, k = 0, 1, ..., λ > 0. Many daily-life events follow the Poisson distribution, e.g., the car accidents that happen each day, the patient visits in the emergency department, etc. The Poisson distribution can be seen as a particular case of the binomial distribution when p is very low and n is very large.
    In each MC round, a large sample size (n = 10000) is used, and each individual is faced with an extremely low accident probability (p= 0.0001). By simulating 100000 MC rounds, we can see that the total number of accidents follows a perfect Poisson distribution.
    """

    def __init__(self, N=100000, n=10000, p=0.0001):
        '''
        Parameters
        ----------
        n,p : the params of binom, i.e., b(n,p)
        '''
        super().__init__("poisson", N)
        self.n = n
        self.p = p

    def run(self, display=True):
        events = scipy.stats.binom.rvs(self.n, self.p, size=self.N)  # directly draw from a b(n,p) dist
        c = collections.Counter(events)

        x_theory = range(min(events), max(events) + 1)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, n=self.n, p=self.p)

        if display:
            super().bar(x=c.keys(), y=c.values(), title="Frequency Histogram\nSampling from b(" + str(self.n) + ',' +
                                                        str(self.p) + '). Simulations = ' + str(self.N))
            super().bar(x=x_theory, y=theory, label=r'$\pi (\lambda=' + str(self.n*self.p) + ')$',
                        title='Theoretical Distribution\n' + r'$\pi (\lambda=' + str(self.n*self.p) + ')$')
