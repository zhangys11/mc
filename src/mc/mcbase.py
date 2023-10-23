import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BARPLOT_KWARGS = {"facecolor": "none", "edgecolor": "black", "alpha": 0.8, "linewidth": 1.1}


class McBase(object):
    """
    dist : 理论分布
    N : Number of experiments.
    """
    def __init__(self, dist, N):
        self.dist = dist
        self.N = N

    def init_theory(self, dist=None, n=0, p=0, x_theory=0, a=0, k=0, df1=0, df2=0):
        if dist is None:
            dist = self.dist

        if dist == "zipf":
            theory = scipy.stats.zipf.pmf(x_theory, a)
        elif dist == "binom":
            theory = scipy.stats.binom.pmf(x_theory, n, p)
        elif dist == 'uniform':
            theory = np.random.uniform(-1, 1, x_theory).mean()
        elif dist == "poisson":
            theory = scipy.stats.poisson.pmf(x_theory, n * p)
        elif dist == "expon":
            theory = scipy.stats.expon.pdf(x=x_theory, scale=round(1 / p + 0.5))
        elif dist == "chi2":
            theory = scipy.stats.chi2.pdf(x=x_theory, df=k)
        elif dist == "t":
            theory = scipy.stats.t.pdf(x=x_theory, df=k)
        elif dist == "f":
            theory = scipy.stats.f.pdf(x=x_theory, dfn=df1, dfd=df2)
        else:
            raise ValueError("Unsupported theoretical distribution: {}".format(self.dist))

        return theory

    def run(self):
        pass

    def bar(self, x=None, y=None, label=None, title=None, draw_points=False):
        plt.figure(figsize=(10, 3))
        # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        if draw_points:
            plt.plot(x, y, 'k+', ms=1, label=label)
            plt.legend()
        plt.bar(x, y, label=label, **BARPLOT_KWARGS)
        plt.title(title)
        plt.show()

    def plot(self, x=None, y=None, label=None, title=None):
        plt.figure(figsize=(10, 3))
        plt.plot(x, y, lw=3, alpha=0.6, c="black", label=label)
        plt.title(title)
        plt.legend()
        plt.show()

    def hist(self, y=None, title=None, density=False):
        plt.figure(figsize=(10, 3))
        plt.hist(y, bins=100, density=density, **BARPLOT_KWARGS)
        plt.title(title)
        plt.show()








