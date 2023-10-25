import collections
import random
import numpy as np
from ..mcbase import McBase


class Exponential(McBase):

    """
    元器件寿命为何符合指数分布？
    定义一个survival game（即每回合有p的死亡率；或电容在单位时间内被击穿的概率）的概率
    取p = 0.001（每回合很小的死亡率），绘制出pmf曲线（离散、等比数组）
    This code defines the probability calculation function of the survival game.
    (e.g. a mortality rate of [p] per turn, or a capacitor having a probability of [p] being broken down per unit
    of time).

    Parameters
    ----------
    num_rounds : survival game rounds
    p : The probability of sudden death / failure / accident per round

    Returns
    -------
    Plot of survival histogram.
    """

    def __init__(self, N=10000, num_rounds=1000, p=0.01):
        super().__init__("expon", N)
        self.num_rounds = num_rounds
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
                                                        str(self.p) + ', players = ' + str(self.N), draw_points=False)
            super().plot(x=x_theory, y=theory, label='θ=' + str(round(1 / self.p + 0.5)),
                         title='Theoretical Distribution\nexponential(θ=' + str(round(1 / self.p + 0.5)) + ')')

        return
