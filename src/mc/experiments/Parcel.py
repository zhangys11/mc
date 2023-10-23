import random
import numpy as np
from ..mcbase import McBase


class Parcel(McBase):

    """
    Simulate a bi-directional parcel passing game.
    [num_players] players form a circle.
    Then, each round the parcel can be passed to the left or right person.
    球回到A手中的试验次数 / 总试验次数 = parcel(试验次数, 玩家数目，每次试验传球次数)

    Parameters
    ----------
    num_players : the number of players.
    num_ops : the number of passes per experiment.

    Returns
    -------
    p : the approximated probability the parcel returns to the starter player.

    Example
    ----
    ### Five people (A, B, C, D, E) stand in a circle to play the game of parcel passing.
    # The rule is that each person can only pass to the neighbor (to the left or to the right).
    # Start the game with A.
    # Q: After 10 passes, what is the probability that the ball will return to A's hand?
    # Use the Monte Carlo method for calculations and compare them with classical probability calculations.

    p = parcel(100000, 5, 10) # simulates 100000 times
    """

    def __init__(self, N=100000, num_players=5, num_ops=10, flavor=1):
        super().__init__(None, N)
        self.num_players = num_players
        self.num_ops = num_ops
        self.flavor = flavor

    def run(self):
        if self.flavor == 1:
            L = 0
            history = []
            for _ in range(self.N):
                position = 0
                for _ in range(self.num_ops):
                    position = (position + random.choice([-1, +1]) + self.num_players) % self.num_players
                history.append(position)
                if position == 0:
                    L += 1
            freq = L / self.N
            print('frequency = {}'.format(freq))
        else:
            L = 0
            history = []
            for _ in range(self.N):
                position = 0
                for _ in range(self.num_ops):
                    seq = list(range(self.num_players))
                    seq.remove(position)
                    position = random.choice(seq)  # pass to any other player
                history.append(position)
                if position == 0:
                    L += 1

            freq = L / self.N
            print('frequency = {}'.format(freq))

            # use graph theory to calculate the theoretical probability
            # construct the affinity matrix
            A = 1 - np.eye(self.num_players)
            M = np.linalg.matrix_power(A, self.num_ops)
            print('Affinity matrix:\n', A)
            print('Affinity matrix powered by num_ops({}): \n{}'.format(self.num_ops, M))
            prob = M[0, 0] / (self.num_players - 1) ** self.num_ops

            print("Probability = {}".format(prob))
            print("MC frequency = {}/{} = {}".format(L, self.N, L / self.N))

        return
