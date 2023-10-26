import numpy as np
import matplotlib.pyplot as plt
from .. import McBase


class Clt(McBase):

    """
    Use MC to demostrate the Central Limit Theorem

    For a population, given an arbitrary distribution.
    Each time from these populations randomly draw [sample_size] samples
    (where [sample_size] takes the value in the [dist]), A total of [N] times.
    The [N] sets of samples are then averaged separately.
    The distribution of these means should be close to normal dist when [sample_size] is big enough.
    """

    def __init__(self, underlying_dist='bernoulli', n=[1, 2, 5, 20], N=10000):
        '''
        Parameters
        ----------
        underlying_dist : base / undeyling /atom distribution. 底层/原子分布
        'uniform' - a uniform distribution U(-1,1) is used.
        'expon' - an exponential distribution Expon(1) is used.
        'poisson' - poisson distribution PI(1) is used.
        'coin' / 'bernoulli' - {0:0.5,1:0.5}
        'tampered_coin' - {0:0.2,1:0.8} # head more likely than tail
        'dice' - {1:1/6,2:1/6,3:1/6,4:1/6,5:1/6,6:1/6}
        'tampered_dice' - {1:0.1,2:0.1,3:0.1,4:0.1,5:0.1,6:0.5} # 6 is more likely
            None - use 0-1 distribution {0:0.5,1:0.5} by default
        n : sample size to be averaged over / summed up.
            Can be an array / list, user can check how the histogram changes with sample size.
        '''
        super().__init__(None, N)
        self.underlying_dist = underlying_dist
        self.sample_size = n

    def run(self):

        rows = len(self.sample_size)
        fig = plt.figure(figsize=(12, rows*3))
        plt.axis('off')

        if self.underlying_dist == 'uniform':

            def f(x):
                return np.random.uniform(-1, 1, x).mean()
            dist_name = "$U(-1,1)$"

        elif self.underlying_dist == 'expon' or self.underlying_dist == 'exponential':

            def f(x):
                return np.random.exponential(scale=1, size=x).mean()
            dist_name = "$Expon(1)$"

        elif self.underlying_dist == 'poisson':

            def f(x):
                return np.random.poisson(lam=1, size=x).mean()
            dist_name = "$\pi(1)$"

        elif self.underlying_dist == 'dice':

            def f(x):
                return np.random.choice(list(range(1, 7)), x).mean()
            dist_name = "PMF {1:1/6,2:1/6,3:1/6,4:1/6,5:1/6,6:1/6}, i.e., dice"

        elif self.underlying_dist == 'tampered_dice':

            def f(x):
                return np.random.choice(list(range(1, 6)) + [6]*5, x).mean()
            dist_name = "PMF {1:0.1,2:0.1,3:0.1,4:0.1,5:0.1,6:0.5}, i.e., a tampered dice"

        elif self.underlying_dist == 'tampered_coin':

            def f(x):
                return np.random.choice([0]+[1]*4, x).mean()
            dist_name = "PMF {0:0.2,1:0.8}, i.e., a tampered coin"

        else:  # dist == 'coin' or 'bernoulli':

            def f(x):
                return np.random.choice([0, 1], x).mean()
            dist_name = "$Bernoulli(0.5), i.e., coin$"

        title = "Use " + dist_name + " to verify CLT (central limit theorem)"
        plt.title(title)

        for row_index, n in enumerate(self.sample_size):

            xbars = []
            for _ in range(self.N):  # MC试验次数
                xbar = f(n)  # np.random.uniform(-1,1,n).mean() #
                xbars.append(xbar)

            ax = fig.add_subplot(rows, 1, row_index + 1)
            # ax.axis('off')
            ax.hist(xbars, density=False, bins=100, facecolor="none", edgecolor="black",
                    label='sample size = ' + str(n))
            ax.legend(loc='upper right')
            ax.set_yticks([])

        # plt.yticks([])
        plt.show()


def clt_all():

    """
    Very the CLT (Central Limit Theorem) with all supported underlying dists
    """

    for underlying_dist in ['uniform', 'expon', 'poisson', 'coin', 'tampered_coin', 'dice', 'tampered_dice']:
        print('-----------', underlying_dist, '-----------')
        clt = Clt(underlying_dist, sample_size=[1, 2, 5, 20, 50], N=10000)
        clt.run()
