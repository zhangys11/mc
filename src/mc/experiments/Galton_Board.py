from ..mcbase import McBase
from ..distributions import Binom


class Galton_Board(McBase):
    def __init__(self, N=5000, num_layers=20, flavor=1):
        super().__init__(None, N)
        self.num_layers = num_layers
        self.flavor = flavor

    def run(self, display=True):
        binom = Binom.Binom(self.N, self.num_layers, self.flavor)
        result = binom.run(display=display)
        return result
