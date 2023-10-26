import numpy as np
import scipy.special
import scipy.stats
from .. import McBase


class F(McBase):

    """
    Create a F-distribution r.v. with [df1] and [df2] degrees of freedom.
    """

    def __init__(self, N=1000, df1=10, df2=10):
        super().__init__("f", N)
        self.df1 = df1
        self.df2 = df2

    def run(self, display=True):

        U = scipy.stats.chi2.rvs(df=self.df1, size=self.N)
        V = scipy.stats.chi2.rvs(df=self.df2, size=self.N)
        Fs = U/self.df1 / (V/self.df2)

        x_theory = np.linspace(round(min(Fs)), round(max(Fs)+0.5), 200)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, df1=self.df1, df2=self.df2)

        if display:
            super().hist(y=Fs, title="Frequency Histogram\ndofs = (" + str(self.df1) + ',' +
                                     str(self.df2) + '). simulations = ' + str(self.N))
            super().plot(x=x_theory, y=theory, label=r'$F (dof1=' + str(self.df1) + ', dof2=' + str(self.df2) + ')$',
                         title='Theoretical Distribution\n' + r'$F (dof1=' + str(self.df1) + ', dof2=' +
                               str(self.df2) + ')$')
