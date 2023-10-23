import numpy as np
from ..mcbase import McBase


class Anova_Stat(McBase):

    """
    验证 ANOVA的F分布假设
    F = MSTR/MSE ~ F(k-1, n-k)
    The H0 assumes mu1=mu2=...=muK.
    In this experiment, all samples are drawn from N(0,1)

    Parameters
    ----------
    k : classes / groups
    n : samples per group. Total sample count is [K]*[n]
    """

    def __init__(self, k=10, n=10, N=10000):
        super().__init__("f", N)
        self.k = k
        self.n = n

    def run(self, display=True):
        FS = []

        for i in range(self.N):

            X = np.random.normal(0, 1, size=(self.n, self.k))
            SSTR = self.n * ((X.mean(axis=0) - X.mean()) ** 2).sum()
            MSTR = SSTR / (self.k - 1)
            SSE = ((X - X.mean()) ** 2).sum()
            MSE = SSE / (self.k * self.n - self.k)  # 此处K*n为公式中n，样本总量
            F = 1.0 * MSTR / MSE
            FS.append(F)

        x_theory = np.linspace(0, np.max(FS), 100)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, df1=self.k-1, df2=self.n*self.k-self.k)

        if display:
            super().hist(
                y=FS, title="Histogram of the ANOVA test statistic ($F = \dfrac{MSTR}{MSE}$)\n. Population is N(0,1)."
                            + str(self.k) + " groups, " + str(self.n) + " samples per group.")
            super().plot(x=x_theory, y=theory, label='$F(' + str(self.k-1) + ',' + str(self.n*self.k-self.k) + ')$',
                         title='Theoretical Distribution\n$F(' + str(self.k-1) + ',' + str(self.n*self.k-self.k) + ')$')

        return
