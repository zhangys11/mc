import collections
import random
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import MultipleLocator
from scipy.special import rel_entr
from tqdm import tqdm
if __package__:
    from . import BARPLOT_KWARGS

class McBase(object):
    def __init__(self, dist):
        self.__theoretical_dist = dist
        self.theory = None
        self.freq = None
        self.x_freq = None
        self.x_theory = None

    def run(self, n=0, N=0, p=0, k=0, df1=0, df2=0, num_clips=0, verbose=False):
        if self.__theoretical_dist == "zipf":
            sets = [1] * num_clips
            for _ in range(N):
                idx1, idx2 = np.random.choice(range(len(sets)), 2, replace=False)
                # print(idx1, idx2)
                sets[idx1] = sets[idx1] + sets[idx2]
                sets.pop(idx2)
            c = collections.Counter(sets)
            vals = np.array(list(c.values()))
            vals = vals / vals.sum()
            self.x_freq = c.keys()
            self.freq = vals
            kld = np.inf
            best_pwr = 1 / 2
            x = list(c.keys())
            num_clips_k = num_clips / N
            for pwr in [1 / 6, 1 / 3, 1 / 2, 2 / 3, 1, 3 / 2]:  # this is an inexact fitting
                a = num_clips_k ** (pwr)
                kldn = sum(rel_entr(vals, scipy.stats.zipf.pmf(x, a)))  # KLD
                if verbose:
                    print("Previous and new KLD between experiment and theory: ", round(kld, 3), round(kldn, 3))
                if kldn < kld:
                    kld = kldn
                    best_pwr = pwr
            a = num_clips_k ** (best_pwr)
            self.x_theory = range(np.array(list(c.keys())).min(), np.array(list(c.keys())).max() + 1)
            self.theory = scipy.stats.zipf.pmf(self.x_theory, a)

        elif self.__theoretical_dist == "binom":
            history = []
            for _ in range(N):
                position = 0  # 初始位置
                for _ in range(n):
                    position = position + random.choice([0, +1])  # 0 left，+1 right
                history.append(position)
            c = collections.Counter(history)
            self.x_freq = c.keys()
            self.freq = c.values()
            self.x_theory = range(n + 1)
            self.theory = scipy.stats.binom.pmf(self.x_theory, n, p)

        elif self.__theoretical_dist =="poisson":
            events = scipy.stats.binom.rvs(n, p, size=N)  # directly draw from a b(n,p) dist
            c = collections.Counter(events)
            self.x_freq = c.keys()
            self.freq = c.values()
            self.x_theory = range(min(events), max(events) + 1)
            self.theory = scipy.stats.poisson.pmf(self.x_theory, n*p)

        elif self.__theoretical_dist =="exponential":
            survival_rounds = []
            for _ in range(N):
                fate = random.choices([0, 1], weights=(1 - p, p), k=n)
                if 1 in fate:
                    survival_rounds.append(fate.index(1))
            c = collections.Counter(survival_rounds)
            self.x_freq = c.keys()
            self.freq = c.values()
            theta = round(1 / p + 0.5)
            self.x_theory = range(np.array(list(c.keys())).min(), np.array(list(c.keys())).max() + 1)
            self.theory = scipy.stats.expon.pdf(x=self.x_theory, scale=theta)

        elif self.__theoretical_dist =="chisq":
            CHISQS = []
            for _ in range(N):
                CHISQS.append(np.sum(np.random.randn(k) ** 2))
            self.freq = CHISQS
            self.x_theory = np.linspace(min(CHISQS), max(CHISQS) + 0.5)
            self.theory = scipy.stats.chi2.pdf(x=self.x_theory, k=k)

        elif self.__theoretical_dist =="student":
            X = np.random.randn(N)
            Y = scipy.stats.chi2.rvs(df=k, size=N)
            ts = X / np.sqrt(Y / k)
            self.freq = ts
            self.x_theory = np.linspace(round(min(ts)), round(max(ts) + 0.5), 200)
            self.theory = scipy.stats.t.pdf(x=self.x_theory, df=k)

        elif self.__theoretical_dist =="F":
            U = scipy.stats.chi2.rvs(df=df1, size=N)
            V = scipy.stats.chi2.rvs(df=df2, size=N)
            Fs = U / df1 / (V / df2)
            self.freq = Fs
            self.x_theory = np.linspace(round(min(Fs)), round(max(Fs) + 0.5), 200)
            self.theory = scipy.stats.f.pdf(x=self.x_theory, dfn=df1, dfd=df2)

        elif self.__theoretical_dist =="benford":
            percents = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
            self.theory = np.array(percents) / 100

        return



    def plot(self, title_freq=None, title_theory=None, plot_freq_class=0, plot_theory_class=0, label=None, display=True):
        if display == True:
            # Frequency
            plt.figure(figsize=(10, 3))
            if plot_freq_class == 0:
                plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
                plt.bar(self.x_freq, self.freq, **BARPLOT_KWARGS)
            else:
                plt.hist(self.freq, bins=100, **BARPLOT_KWARGS)
            plt.title(title_freq)
            plt.show()

            # Theory
            plt.figure(figsize=(10, 3))
            if plot_theory_class == 0:
                plt.bar(self.x_theory, self.theory, label=label, **BARPLOT_KWARGS)
                ax = plt.gca()
                ax.xaxis.set_major_locator(MultipleLocator(1))  # use interger ticks
            elif plot_theory_class == 1:
                plt.plot(self.x_theory, self.theory, label=label)
                plt.legend()
            else:
                plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
                plt.plot(self.x_theory, self.theory, 'k+', ms=1, label=label)
                plt.legend()
                plt.bar(self.x_theory, self.theory, **BARPLOT_KWARGS)
            plt.title(title_theory)
            plt.show()

    def stat_plot(self, x=0, result1=[], result2=None, y=0, bins=0, title_freq_1=None, title_freq_2=None, title_theory=None, label=None):
        plt.figure()
        plt.hist(result1, density=False, bins=bins, **BARPLOT_KWARGS)
        plt.title(title_freq_1)  # $b("+ str(K) +", 1/2)$
        plt.show()

        if result2:
            plt.figure()
            plt.hist(result2, density=False, bins=bins, **BARPLOT_KWARGS)
            plt.title(title_freq_2)  # $b("+ str(K) +", 1/2)$
            plt.show()

        plt.figure()
        plt.plot(x, y, lw=3, alpha=0.6, c="black", label=label)
        plt.title(title_theory)
        plt.legend()
        plt.show()








