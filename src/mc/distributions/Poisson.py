import collections
import scipy.special
import scipy.stats
from ..mcbase import McBase


class Poisson(McBase):

    """
    possion 是 b(n,p), n很大，p很小的一种极限分布
    假设一个容量为n的群体，每个个体发生特定事件（如意外或事故）的概率为p（极低），那么总体发生事件的总数近似符合泊松
    """

    def __init__(self, N=100000, n=10000, p=0.0001):
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

        return
