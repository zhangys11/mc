import numpy as np
from tqdm import tqdm
from ..mcbase import McBase


class Bartlett_Stat(McBase):

    """
    Bartlett's test is used to test homoscedasticity, that is, if multiple samples are from populations with equal
    variances. Verify the Bartlett's Test statistic is a X2 random variable.

    Parameters
    ----------
    k : groups / classes
    n : samples per class. In this experiment, all group sizes are equal.
    """

    def __init__(self, k=5, n=10, N=1000):
        super().__init__('chi2', N)
        self.k = k
        self.n = n

    def run(self, display=True):
        BTs = []
        for _ in tqdm(range(self.N)):
            X = np.random.randn(self.k, self.n)
            Si_2 = np.var(X, axis=1)
            SP_2 = (1 / (self.n*self.k - self.k)) * sum([i * (self.n - 1) for i in Si_2])
            ln_Si2 = np.log([i for i in Si_2])
            BT = ((self.n*self.k - self.k) * np.log(SP_2) - sum([i * (self.n - 1) for i in ln_Si2])) / \
                 (1 + (1 / (3 * (self.k - 1))) * (self.k * ((1 / self.n) - 1 / (self.n*self.k - self.k))))
            BTs.append(BT)

        x_theory = np.linspace(np.min(BTs), np.max(BTs), 100)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, k=self.k-1)

        if display:
            super().hist(
                y=BTs,
                title="Histogram of Bartlett's $\chi^2$ statistic ($\chi^2 = \dfrac{(N-k)\ln^{(S_{P}^2)}-\sum_{i=1}^{k}\
                (n_{i}-1)\ln^{(S_{i}^2)}}{1+\dfrac{1}{3(k-1)}(\sum_{i=1}^{k}(\dfrac{1}{n_{i}})-\dfrac{1}{N-k})}$)")
            super().plot(x=x_theory, y=theory, label='dof = ' + str(self.k - 1),
                         title='Theoretical Distribution \n $\chi^2(dof=' + str(self.k-1) + ')$')

        return
