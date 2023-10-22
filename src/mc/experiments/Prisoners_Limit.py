import math
import numpy as np
import matplotlib.pyplot as plt
from ..mcbase import McBase
from .Prisoners import Prisoners


class Prisoners_Limit(McBase):

    """
    Test how the survival rate changes with n. The limit is 1-ln2.

    Parameters
    ----------
    ns : prisoner numbers to be tested.
    repeat : repeat multiple times to calculate the SD (standard deviation)
    SD : how many SD (standard deviation) to show in the error bar chart
    """

    def __init__(self, ns=[10, 20, 30, 40, 50, 100, 200, 500, 1000], repeat=10, SD=1, N=1000):
        super().__init__(None, N)
        self.ns = ns
        self.repeat = repeat
        self.SD = SD

    def run(self):
        fss = []

        if self.repeat is None or self.repeat < 1:
            self.repeat = 1

        for _ in range(self.repeat):
            fs = []
            for n in self.ns:
                prisoners = Prisoners(n, N=self.N)
                fs.append(prisoners.run())
            fss.append(fs)

        fss = np.array(fss)  # repeat-by-ns matrix
        fsm = fss.mean(axis=0)

        plt.figure(figsize=(10, 4))

        if self.SD == 0 or self.repeat <= 1:
            plt.scatter(self.ns, fsm, label='survival chance')
            plt.plot(self.ns, fsm)
        else:  # show +/- std errorbar
            plt.errorbar(self.ns, fsm, fss.std(axis=0)*self.SD,
                         # color = ["blue","red","green","orange"][c],
                         linewidth=1,
                         alpha=0.2,
                         label='survival chance. mean ± ' + str(self.SD) + ' SD',
                         )
            plt.scatter(self.ns, fsm)

        plt.hlines(y=1 - math.log(2), xmin=np.min(self.ns), xmax=np.max(self.ns),
                   color='r', label=r'$1- ln(2) = $' + str(round(1 - math.log(2), 3)))
        plt.xlabel('prisoner number')
        plt.legend()
        plt.show()
