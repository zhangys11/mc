from ..mcbase import McBase


class Levene_Test(McBase):

    """
    Levene's test is used to test if k samples have equal variances.
    Verify the Levene's Test statistic (W) is a X2 random variable.

    Parameters
    ----------
    n : samples per class. In this experiment, all group sizes are equal.
    k : groups / classes
    """

    def __init__(self, n=5, k=2, N=1000):
        super().__init__(None, N)
        self.n = n
        self.k = k

    def run(self, display=True):
        raise NotImplementedError
