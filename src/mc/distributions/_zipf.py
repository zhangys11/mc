import collections
import numpy as np
from scipy.special import rel_entr
import scipy.special
import scipy.stats
from .. import McBase


class Zipf(McBase):

    """
    The Zipf law / distribution is published in 1949 by Harvard linguist George Kingsley Zipf.
    It can be expressed as follows: in the corpus of natural language,
    the frequency of a word appears inversely proportional to its ranking
    in the frequency table.
    The experiment associated with this law is the paper clip experiment,
    that is, two paper clips are randomly drawn and connected together,
    then put back, and then do the above again.
    After enough rounds, the clips of different lengths will obey zipf distribution.

    Note
    ----
    Internally, we use grid search via the KLD metric to determine the best-fit zipf dist.
    """

    def __init__(self, N=10000, n=16000):
        '''        
        Parameters
        ----------
        n : The total number of paper clips. It should always be greater than N, i.e., num_rounds.
        '''
        super().__init__("zipf", N)
        self.num_clips = n

    def run(self, verbose=False, display=True):
        if self.num_clips <= self.N:  # clip总数，应大于抽样拼接次数，
            print('Error: num_clips must > num_rounds')
            return

        sets = [1]*self.num_clips
        for _ in range(self.N):
            idx1, idx2 = np.random.choice(range(len(sets)), 2, replace=False)
            # print(idx1, idx2)
            sets[idx1] = sets[idx1] + sets[idx2]
            sets.pop(idx2)
        c = collections.Counter(sets)
        vals = np.array(list(c.values()))
        vals = vals / vals.sum()

        kld = np.inf
        best_pwr = 1/2
        x = list(c.keys())
        num_clips_k = self.num_clips / self.N
        for pwr in [1/6, 1/3, 1/2, 2/3, 1, 3/2]:  # this is an inexact fitting
            a = num_clips_k ** pwr
            kldn = sum(rel_entr(vals, scipy.stats.zipf.pmf(x, a)))  # KLD
            if verbose:
                print("Previous and new KLD between experiment and theory: ",
                      round(kld, 3), round(kldn, 3))
            if kldn < kld:
                kld = kldn
                best_pwr = pwr
        a = num_clips_k ** best_pwr
        x_theory = range(np.array(list(c.keys())).min(), np.array(list(c.keys())).max() + 1)
        theory = super().init_theory(dist=self.dist, x_theory=x_theory, a=a)

        if display:
            super().bar(x=c.keys(), y=vals, title="Frequency Histogram\nclips=" + str(self.num_clips) + ", rounds=" +
                                                  str(self.N), draw_points=False)
            super().bar(x=x_theory, y=theory, title='Theoretical Distribution\nzipf(α = '+str(round(a, 2)) + ')',
                        draw_points=False)
