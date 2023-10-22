import numpy as np
import matplotlib.pyplot as plt
from ..mcbase import McBase


class Pi_Illustrate(McBase):

    """
    Generate an illustration plot for the flavor 2 pi experiment
    """

    def __init__(self, N=2000):
        super().__init__(None, N)

    def run(self):
        r = np.round(self.N / 500 + 0.5)

        xs = np.random.uniform(-r, r, self.N)
        ys = np.random.uniform(-r, r, self.N)

        plt.figure(figsize=(4*r, 4*r))
        draw_circle = plt.Circle((0., 0.), r, fill=False)
        plt.gcf().gca().add_artist(draw_circle)
        plt.scatter(xs, ys, s=round(self.N/50 + 0.5), marker='o',
                    edgecolor='gray', facecolor='none')
        plt.axis("square")
        plt.axis("off")
        plt.xticks([-r, 0, r])
        plt.yticks([-r, 0, r])
        plt.show()
