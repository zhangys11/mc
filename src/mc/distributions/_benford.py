import collections.abc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

if __package__:
    from .. import DATA_FOLDER, BARPLOT_KWARGS
    from .. import McBase


class Benford(McBase):

    """
    Benford's law: also called the Newcomb–Benford law, the law of anomalous numbers, or the first-digit law, is an
    observation about the frequency distribution of leading digits in many real-life sets of numerical data. The law
    states that in many naturally occurring collections of numbers, the leading significant digit is likely to be small.
    
    The PMF (Probability Mass Function) of Benford: 

    d 1 2 3 4 5 6 7 8 9
    p 30.1% 17.6% 12.5% 9.7% 7.9% 6.7% 5.8% 5.1% 4.6%
    
    Benford's law applies to a wide variety of data sets, including electricity bills, street addresses, stock prices, population numbers, death rates, lengths of rivers, physical and mathematical constants, and processes described by power laws (which are very common in nature).
    
    Note
    ----
    This class support real-life stock datasets and the fibonacci series as input. Fibonacci represents how a population (e.g., rabbits) grows in a resource-unlimited environment.
    In intuitive explanation of why it follows Benford: at a steady breeding speed, it takes much longer time from 1000 to 2000 (need to increase by 1000) than from 900 to 1000 (only need to increase by 100). Therefore, it stays longer at leading digit 1 than other digits.
    """

    def __init__(self, N=1000, data="stock"):
        '''
        Parameters
        ----------
        data : data set / experiment to use
            'fibonacci' - use the top-N fibonacci series
            'stock' - use 20-year stock trading volume data of Apple Inc. (AAPL)
            'trade' - annual trade data for countries. https://comtrade.un.org/data/mbs
            list or array-like - directly use the passed data 
        N : only used for 'fibonacci'

        Note
        ----
        If for some reason, the AAPL.csv is missing, use the following code to retrieve:

            import yfinance as yf
            data = yf.download('AAPL','2000-01-01','2020-05-01') # may also try 'GOOG', etc.
            data.to_csv('AAPL.csv')
        '''

        super().__init__(None, N)
        self.data = data

    def run(self, display=True):
        volumes = []
        title = ''
        if self.data == 'stock':
            volumes = pd.read_csv(DATA_FOLDER + '/AAPL.csv')['Volume'].values
            title = '20-year stock trading volume of AAPL'
        elif self.data == 'trade':
            volumes = pd.read_csv(DATA_FOLDER + '/MBSComtrade.csv')['value'].values
            title = 'Annual trade for countries (UN Comtrade Database)'
        elif self.data == 'fibonacci':
            phi = (1 + 5 ** 0.5) / 2.0
            volumes = np.round((np.power(phi, range(1, self.N+1)) - np.power(1-phi, range(1, self.N+1))) / 5**0.5)
            title = 'Fibonacci series (Top-' + str(self.N) + ')'
        elif isinstance(self.data, collections.abc.Sequence): # isinstance(self.data, list):
            volumes = np.array(self.data)
            title = str(len(volumes)) + ' data points'

        cnts = np.zeros(10)

        for v in volumes:
            s = str(v).lstrip('0.')
            if s and len(s) > 0:
                leading_digit = int(s[0])
                cnts[leading_digit] += 1

        total = np.sum(cnts)

        if display:
            plt.figure(figsize=(10, 3))
            plt.bar(range(1, len(cnts)), height=cnts[1:], **BARPLOT_KWARGS)
            for a, b in zip(range(1, len(cnts)), cnts[1:]):
                plt.text(a, b, '%.0f %%' % (b*100/total), ha='center', va='bottom')

            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.title("Frequency histogram of the leading digits\n" + title)

            plt.figure(figsize=(10, 3))
            plt.title('Theoretical Distribution\nBenford PMF')
            percents = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
            percents = np.array(percents) / 100
            plt.bar(range(1, 10), height=percents, label='Benford PMF', **BARPLOT_KWARGS)
            for a, b in zip(range(1, 10), percents):
                plt.text(a, b, '%.1f %%' % (b*100), ha='center', va='bottom')

            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.legend()
            plt.show()
