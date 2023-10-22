import random
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

    def __init__(self, N=100000, num_players=5, num_ops=10):
        super().__init__(None, N)
        self.num_players = num_players
        self.num_ops = num_ops

    def run(self):
        L = 0
        history = []
        for _ in range(self.N):
            position = 0
            for _ in range(self.num_ops):
                position = (position + random.choice([-1, +1]) + self.num_players) % self.num_players
            history.append(position)
            if position == 0:
                L += 1

        return L/self.N
