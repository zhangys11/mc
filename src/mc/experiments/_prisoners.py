import numpy as np
import math
import matplotlib.pyplot as plt
from ..mcbase import McBase


class Prisoners(McBase):

    """
    The famous the locker puzzle.
    We will prove that the limit is (1-ln2) when n approaches +inf

    Parameters
    ----------
    n : how many prisoners

    Returns
    -------
    frequency : of the N experiments, how many times the n prisoners survive.
    """

    def __init__(self, n=100, N=2000):
        super().__init__(None, N)
        self.n = n

    def run(self):

        WINS = 0
        FAILS = 0
        for i in range(self.N):  # i is MC experiment NO, we will not use it later.

            boxes_inner = np.random.choice(list(range(self.n)), self.n, replace=False)
            failed = False  # at least one prisoner failed the test

            for j in range(self.n):  # j is prisoner NO
                found = False

                target_box_index = j
                for k in range(round(self.n/2)):  # k is the draw round
                    target_box_index = boxes_inner[target_box_index]
                    if target_box_index == j:
                        found = True
                        break

                if found is False:
                    # DOOMED
                    failed = True
                    break

            if failed:
                FAILS += 1
            else:
                WINS += 1

        return WINS / (WINS + FAILS)


def asymptotic_analysis(ns=[10, 20, 30, 40, 50, 100, 200, 500, 1000], repeat=10, SD=1, N=1000):

    """
    Test how the survival rate changes with n. The limit is 1-ln2.

    Parameters
    ----------
    ns : prisoner numbers to be tested.
    repeat : repeat multiple times to calculate the SD (standard deviation)
    SD : how many SD (standard deviation) to show in the error bar chart
    """
    fss = []

    if repeat is None or repeat < 1:
        repeat = 1

    for _ in range(repeat):
        fs = []
        for n in ns:
            prisoners = Prisoners(n, N=N)
            fs.append(prisoners.run())
        fss.append(fs)

    fss = np.array(fss)  # repeat-by-ns matrix
    fsm = fss.mean(axis=0)

    plt.figure(figsize=(10, 4))

    if SD == 0 or repeat <= 1:
        plt.scatter(ns, fsm, label='survival chance')
        plt.plot(ns, fsm)
    else:  # show +/- std errorbar
        plt.errorbar(ns, fsm, fss.std(axis=0)*SD,
                     # color = ["blue","red","green","orange"][c],
                     linewidth=1,
                     alpha=0.2,
                     label='survival chance. mean ± ' + str(SD) + ' SD',
                     )
        plt.scatter(ns, fsm)

    plt.hlines(y=1 - math.log(2), xmin=np.min(ns), xmax=np.max(ns),
               color='r', label=r'$1- ln(2) = $' + str(round(1 - math.log(2), 3)))
    plt.xlabel('prisoner number')
    plt.legend()
    plt.show()
