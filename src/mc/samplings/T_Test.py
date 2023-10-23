import numpy as np
from ..mcbase import McBase


class T_Stat(McBase):

    """
    Sample [n] samples from a normal distribution, and compute the t statistic follows the student's distribution.

    Parameters
    ----------
    n : samples
    """

    def __init__(self, n=10, N=10000):
        super().__init__("t", N)
        self.n = n

    def run(self, display=True):
        ts = []

        for i in range(self.N):
            # 添加for循环
            X = np.random.normal(0, 1, size=self.n)
            T = (X.mean() - 0) / X.std() * np.sqrt(self.n)
            ts.append(T)

        x_theory = np.linspace(np.min(ts), np.max(ts), 100)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.n-1)

        if display:
            super().hist(y=ts,
                         title="Histogram of the test statistic ($t = \dfrac{X\u0305 -\mu}{S/\sqrt{n}}$).\n \
                         Population is N(0,1). " + str(self.n) + " samples.")
            super().plot(x=x_theory, y=theory, label='$t (dof=' + str(self.n-1) + ')$',
                         title='Theoretical Distribution')

        return
