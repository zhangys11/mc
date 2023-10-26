import collections
import random
import numpy as np
from .. import McBase


class Exponential(McBase):

    """
    This class defines a survival game, i.e., each round has p death rate.
    p is very small. p is per-turn mortality rate, or a capacitor having a probability of being broken down per unit of time).     
    We define this survival game to illustrate the underlying mechanism of the exponential distribution. Because the sudden death game can approximate many real-life accidents or electronic component failures (e.g., capacity breakdown or LCD pixel defect), the resulting exponential distribution can be used in survival analysis and lifespan estimation. In each round of this survival game, the test subject (player) is faced with a very low sudden death probability (p).
    If you choose p = 0.001 and simulate 10,000 MC rounds. The generated histogram is very close to the exponential distribution. This function can be used to illustrate the generation mechanism of the exponential distribution.
    
    p(x) = $ 1/\theta * exp(-x/\theta) $ , if x > 0
    """

    def __init__(self, N=10000, n=1000, p=0.01):
        '''
        Parameters
        ----------
        n : survival game rounds
        p : The probability of sudden death / failure / accident per round
        '''
        super().__init__("expon", N)
        self.num_rounds = n
        self.p = p

    def run(self, display=True):
        survival_rounds = []
        for _ in range(self.N):
            fate = random.choices([0, 1], weights=(1-self.p, self.p), k=self.num_rounds)
            if 1 in fate:
                survival_rounds.append(fate.index(1))
            # else: # still lives, i.e., > num_rounds
            #     survival_rounds.append(num_rounds)

        c = collections.Counter(survival_rounds)
        x_theory = range(np.array(list(c.keys())).min(), np.array(list(c.keys())).max() + 1)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, p=self.p)

        if display:
            super().bar(x=c.keys(), y=c.values(), title="Frequency Histogram\nper-round sudden death probability p=" +
                                                        str(self.p) + ', game rounds = ' + str(self.num_rounds) + ', simulations = ' + str(self.N), draw_points=False)
            super().plot(x=x_theory, y=theory, label='θ=' + str(round(1 / self.p + 0.5)),
                         title='Theoretical Distribution\nexponential(θ=' + str(round(1 / self.p + 0.5)) + ')')
