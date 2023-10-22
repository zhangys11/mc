from ..mcbase import McBase
from ..distributions import Exponential


class Sudden_Death(McBase):
    def __init__(self, N=10000, num_rounds=1000, p=0.01):
        super().__init__(None, N)
        self.num_rounds = num_rounds
        self.p = p

    def run(self, display=True):
        exponential = Exponential.Exponential(self.N, self.num_rounds, self.p)
        exponential.run(display=display)
        return
