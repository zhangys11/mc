import collections
import random
from ..mcbase import McBase


class Binom(McBase):

    """
    The Galton board is a physical model of the binomial distribution.
    When samples are sufficient, you can also observe CLT.
    If there are [num_layers] layers of nail plates, the number of nails in each layer increases from the beginning one
    by one, And the nail plates have [num_layers+1] corresponding grooves under them.
    This function solves the probability (N times) for a ball falling into each slot by using Monte Carlo's algorithm.

    Parameters
    ----------
    num_layers : The number of nail plate layers.
    flavor : 1 or 2. Which implementation to use.

    Returns
    -------
    A [num_layers+1] long vector : Freqency Historgm, i.e., the number of balls that fall into each slot.
    """

    def __init__(self, N=5000, num_layers=20, flavor=1):
        super().__init__("binom", N)
        self.num_layers = num_layers
        self.flavor = flavor

    def run(self,  display=True):
        result = [0 for i in range(self.num_layers + 1)]
        if self.flavor == 1:
            for _ in range(self.N):
                pos = 0
                for _ in range(self.num_layers):
                    if random.random() > 0.5:
                        pos += 1
                result[pos] += 1

            x_freq = range(self.num_layers+1)
            freq = result

        else:
            history = []
            for _ in range(self.N):
                position = 0  # 初始位置
                for _ in range(self.num_layers):
                    position = position + random.choice([0, +1])  # 0 向左落，+1 向右落
                history.append(position)
            c = collections.Counter(history)
            for pair in zip(c.keys(), c.values()):
                result[pair[0]] = pair[1]

            x_freq = c.keys()
            freq = c.values()

        x_theory = range(self.num_layers+1)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, n=self.num_layers, p=0.5)

        if display:
            super().bar(x=x_freq, y=freq, title="Frequency Histogram\n" + "(" + "layers=" + str(self.num_layers) + ", balls=" +
                                                str(self.N) + ")", draw_points=False)
            super().bar(x=x_theory, y=theory, label='b (' + str(self.num_layers) + ',' + str(0.5) + ')',
                        title='Theoretical Distribution\nbinomial(n=' + str(self.num_layers) + ',p=' + str(0.5) + ')',
                        draw_points=True)

        return result
