import numpy as np
import math
import matplotlib.pyplot as plt
from .. import McBase


class Prisoners(McBase):

    """
    The famous the locker puzzle.
    We will prove that the limit is (1-ln2) when n approaches +inf

    The puzzle
    ----------
    The hundred-prisoner puzzle or the locker puzzle was first addressed by Danish scientist Peter Bro Miltersen (Gál and Miltersen 2007) (Warshauer and Curtin 2006). In this puzzle, there are 100 lockers containing No.1 to No.100. In each round, one prisoner will open 50 lockers. The game will continue if his/her number is found inside any of the opened lockers. Otherwise, the game is over, and all prisoners will be executed. The prisoners cannot communicate with each other during the game.
    What are the best strategy and best survival probability?

    Pure guess survial rate is (1/2)^100, nearly 0.  
    The best strategy is the circular chain. i.e., the prisoner first opens the locker of his or her number, then opens the locker whose number is inside the last locker. With this strategy, the survival probability equals the probability of creating circular chains no longer than 50. This probablity is about 0.3118.
    """

    def __init__(self, n=100, N=2000):
        '''
        Parameters
        ----------
        n : how many prisoners
        '''
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

        self.freq = WINS / (WINS + FAILS)
        print(f'Observed survival rate = {WINS}/{WINS+FAILS} = {self.freq}')


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
                obj = Prisoners(n, N=N)
                obj.run()
                fs.append(obj.freq)
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
