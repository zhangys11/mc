import numpy as np
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
