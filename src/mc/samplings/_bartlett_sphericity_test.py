from .. import McBase


class Bartlett_Sphericity_Test(McBase):

    """
    NOte
    ----
    Bartlett’s Test of Sphericity compares an observed correlation matrix to
    the identity matrix. Essentially it checks to see if there is a certain redundancy
    between the variables that we can summarize with a few number of factors.
    This test is used as a precursory test for Factor Analysis or PCA.
    """

    def __init__(self, N=10000):
        super().__init__(None, N)

    def run(self, display=True):
        raise NotImplementedError
