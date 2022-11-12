import collections
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import MultipleLocator
from scipy.special import rel_entr
from scipy import stats
import scipy.special
import scipy.stats
if __package__:
    from . import DATA_FOLDER
else:
    import os.path
    DATA_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/data/"

def zipf(num_rounds = 10000, num_clips_k = 1.6, verbose = False):
    
    """
    The Zipf law / distribution is published in 1949 by Harvard linguist George Kingsley Zipf. 
    It can be expressed as follows: in the corpus of natural language, 
    the frequency of a word appears inversely proportional to its ranking 
    in the frequency table. 
    The experiment associated with this law is the paper clip experiment, 
    that is, two paper clips are randomly drawn and connected together, 
    then put back, and then do the above again. 
    After enough rounds, the clips of different lengths will obey zipf distribution. 
    Parameters
    ----------
    num_rounds : The number of random samples. 抽样拼接次数
    num_clips_k : The total number of paper clips should be greater than [num_rounds]. This is the ratio of the numbers. 
    Should always > 1. Some of values are 1.6, 1.8, 2, 2.5, 3.
    Note
    ----
    Internally, we use grid search via the KLD metric to determine the best-fit zipf dist. 
    """ 
    
    if num_clips_k <= 1:
        print('Error: num_clips_k must > 1')
        return

    num_clips = int(num_rounds*num_clips_k) # clip总数，应大于抽样拼接次数，i.e., k > 1
    history = []
    sets = [1]*num_clips    
    for iter in range(num_rounds):
        idx1, idx2 = np.random.choice(range( len(sets)), 2, replace = False)
        # print(idx1, idx2)
        sets[idx1] = sets[idx1] + sets[idx2]
        sets.pop(idx2)
    c = collections.Counter(sets)
    plt.figure(figsize = (10,3))
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    vals = np.array(list(c.values())) 
    vals = vals / vals.sum()
    plt.bar(c.keys(), vals, color = 'gray', edgecolor='black')
    plt.title("Frequency Histogram\nclips=" + str(num_clips) + ", rounds=" + str(num_rounds) + ", k=" + str(num_clips_k))
    plt.show()

    kld = np.inf
    best_pwr = 1/2
    x = list(c.keys()) 

    for pwr in [1/6, 1/3, 1/2, 2/3, 1, 3/2]: # this is an inexact fitting

        a = num_clips_k ** (pwr)
        kldn = sum(rel_entr(vals, stats.zipf.pmf(x, a))) # KLD

        if verbose:
            print("Previous and new KLD between experiment and theory: ", round(kld,3), round(kldn,3) )
        if kldn < kld:
            kld = kldn
            best_pwr = pwr
    
    a = num_clips_k ** (best_pwr)
    
    plt.figure(figsize = (10,3))
    x = range(np.array(list(c.keys())).min(), np.array(list(c.keys())).max() + 1) # np.arange(zipf.ppf(0.01, a), zipf.ppf(0.99, a)) 
    plt.bar(x, stats.zipf.pmf(x, a), color = 'gray', edgecolor='black')
    # plt.plot(x, zipf.pmf(x, a), 'bo', ms=8, label='zipf pmf')
    # plt.vlines(x, 0, zipf.pmf(x, a), colors='b', lw=5, alpha=0.5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1)) # use interger ticks
    plt.title('Theoretical Distribution\nzipf(alpha = '+str(round(a,2)) + ')')
    plt.show()
    
    
def binom(num_layers = 20, N = 5000, flavor = 1, display = True):

    """
    The Galton board is a physical model of the binomial distribution. 
    When samples are sufficient, you can also observe CLT. 
    If there are [num_layers] layers of nail plates, the number of nails in each layer increases from the beginning one by one, 
    And the nail plates have [num_layers+1] corresponding grooves under them. 
    This function solves the probability (N times) for a ball falling into each slot by using Monte Carlo's algorithm.
    Parameters
    ----------
    num_layers : The number of nail plate layers.
    N : Number of experiments.
    flavor : 1 or 2. Which implementation to use.
    Returns
    -------
    A [num_layers+1] long vector : Freqency Historgm, i.e., the number of balls that fall into each slot.
    """

    if display:
        plt.figure(figsize = (10,3))
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

    result = [0 for i in range(num_layers + 1)]

    if flavor == 1:        
        for i in range (N):
            pos = 0
            for j in range (num_layers):
                if random.random() > 0.5:
                    pos += 1
            result [pos] += 1
        
        if display:
            plt.bar(range(num_layers+1), result, color = 'gray', linewidth=1.2, edgecolor='black')

    else:

        history = []
        for iter in range(N):
            position = 0 # 初始位置
            for layer in range(num_layers):
                position = position + random.choice([0, +1]) # 0 向左落，+1 向右落
            history.append(position)
        c = collections.Counter(history)
        for pair in zip(c.keys(), c.values()):
            result[pair[0]] = pair[1]

        if display:
            plt.bar(c.keys(), c.values(), color = 'gray', linewidth=1.2, edgecolor='black')
        
    if display:
        plt.title("Frequency Histogram\nlayers=" + str(num_layers) + ", balls=" + str(N) +")")
        plt.show()

        plt.figure(figsize = (10,3))
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

        n = num_layers
        p = 0.5    

        x = range(num_layers+1)
        plt.plot(x, stats.binom.pmf(x, n, p), 'k+', ms=1, label='b (' + str(n) + ',' + str(p) + ')')
        plt.legend()
        plt.title('Theoretical Distribution\nbinomial(n='+str(n) + ',p='+ str(p) + ')')
        plt.bar(x, stats.binom.pmf(x, n, p), color='gray', linewidth=1.2, edgecolor='black')
        plt.show()

    return result
    

def poisson(n = 10000, p = 0.0001, N = 100000):
    '''
    possion 是 b(n,p), n很大，p很小的一种极限分布
    假设一个容量为n的群体，每个个体发生特定事件（如意外或事故）的概率为p（极低），那么总体发生事件的总数近似符合泊松
    '''
    events = stats.binom.rvs(n, p, size=N) # directly draw from a b(n,p) dist
    c = collections.Counter(events)

    plt.figure(figsize = (10,3))
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.bar(c.keys(), c.values(), color = 'gray', linewidth=1.2, edgecolor='black')
    plt.title("Frequency Histogram\nSampling from b(" + str(n) + ',' + str(p) + '). \
Simulations = ' + str(N))
    plt.show()

    plt.figure(figsize = (10,3)) 
    plt.title('Theoretical Distribution\n' + r'$\pi (\lambda='+ str(n*p) + ')$')
    x = range(min(events), max(events) + 1) 
    plt.bar(x,stats.poisson.pmf(x, n*p), color='gray', linewidth=1.2, edgecolor='black', \
        label = r'$\pi (\lambda='+ str(n*p) + ')$')
    plt.legend()
    plt.show()

def exponential(num_rounds = 1000, p = 0.01, N = 10000):
    """
    元器件寿命为何符合指数分布？  
    定义一个survival game（即每回合有p的死亡率；或电容在单位时间内被击穿的概率）的概率
    取p = 0.001（每回合很小的死亡率），绘制出pmf曲线（离散、等比数组）
    This code defines the probability calculation function of the survival game.
    (e.g. a mortality rate of [p] per turn, or a capacitor having a probability of [p] being broken down per unit of time).
    Parameters
    ----------
    num_rounds : survial game rounds
    p : The probability of suddent death / failure / accident for each round
    N : players / sample count / MC simulations
    
    Returns
    -------
    Plot of survival histogram.
    """

    survival_rounds = []
    for player in range(N):
        fate = random.choices([0,1], weights=(1-p,p), k = num_rounds)
        if 1 in fate:
            survival_rounds.append(fate.index(1))
        # else: # still lives, i.e., > num_rounds
        #     survival_rounds.append(num_rounds)
            
    c = collections.Counter(survival_rounds)

    plt.figure(figsize = (10,3))
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.bar(c.keys(), c.values(), color = 'gray', edgecolor='black')
    plt.title("Frequency Histogram\nper-round sudden death probality p=" + str(p) + ', players = ' + str(N))
    plt.show()

    '''
    # survival game.It has survived n rounds; dies in n+1 round. 
    def survival_dist(n,p):
        return pow((1-p),n)*p
    plt.figure(figsize = (10,3))
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    x = linspace(0,num_rounds,num_rounds+1)
    plt.plot(x, survival_dist(x,p))
    plt.title("Theoretical Distribution\nsurvival PMF" )
    plt.show()
    '''

    plt.figure(figsize = (10,3))    
    theta = round(1 / p + 0.5)
    plt.title('Theoretical Distribution\nexponential(θ='+ str(theta) + ')')  
    # pmf: Probability mass function. i.e. pdf
    # ppf: Percent point function (inverse of cdf — percentiles).
    # x = np.arange(stats.expon.ppf(q=0.001, scale=theta), stats.expon.ppf(q=0.999, scale=theta))
    x = range(np.array(list(c.keys())).min(), np.array(list(c.keys())).max() + 1) 
    plt.plot(x,stats.expon.pdf(x=x, scale=theta))
    # plt.show();
    # plt.plot(x,expon.cdf(x=x, scale=s))
    # plt.plot(x,expon.sf(x=x, scale=s)) # when s = 1, sf and pdf overlaps
    plt.show() 


def chisq_pdf(x, k):
    '''
    The theoretical PDF of CHISQ dist. 
    This is a direct implementation based on the definition.
    Alternately, we may use scipy.stats.chi2

    Paramters
    ---------
    k : dof, degree of freedom
    '''

    return x**(k/2-1)*np.exp(-x/2)/(2**(k/2)*scipy.special.gamma(k/2))
    

def chisq_pdf_dist(ul=0, ub=10, k=2, flavor = 1):
    '''
    Parameters
    ----------
    flavor : 
        1 - use self implementation
        2 - use scipy.stats.chi2
    '''
    
    pdf = chisq_pdf if flavor == 1 else scipy.stats.chi2.pdf

    plt.figure(figsize = (10,3))
    plt.title('Theoretical Distribution\n' + r'$\chi^2(dof='+ str(k) + ')$')  
    plt.plot(np.linspace(ul,ub), pdf(np.linspace(ul,ub),k),\
        lw=3, alpha=0.6, label = r'$\chi^2(dof='+ str(k) + ')$')
    plt.legend()
    plt.show()     

def chisq(k=10, N = 10000):
    '''
    The squared sum of [k] r.v.s. from standard normal distributions is a chisq statistic.
    This function will verify it via [N] MC experiments.
    [k]个 N(0,1)^2 r.v.s. 的和为一个卡方分布的统计量

    Parameters
    ----------
    k : How many r.v.s. to use
    N : MC simulations
    '''

    CHISQS = []

    for i in range(N):
        CHISQS.append( np.sum(np.random.randn(k)**2) )  

    plt.figure(figsize = (10,3))
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.hist(CHISQS, bins=100, color = 'gray', edgecolor='black')
    plt.title("Frequency Histogram\ndegree of freedom =" + str(k) + ', simulations = ' + str(N))
    plt.show()

    ul=min(CHISQS)
    ub=max(CHISQS)+0.5
    chisq_pdf_dist(round(ul), round(ub), k=k)

def student(k=5, N = 10000):
    '''
    The t-distribution
    '''
    X = np.random.randn(N)
    Y = scipy.stats.chi2.rvs(df=k, size=N)

    ts = X/np.sqrt(Y/k)

    plt.figure(figsize = (10,3))
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.hist(ts, bins=100, color = 'gray', edgecolor='black')
    plt.title("Frequency Histogram\ndegree of freedom =" + str(k) + ', simulations = ' + str(N))
    plt.show()

    plt.figure(figsize = (10,3)) 
    plt.title('Theoretical Distribution\n' + r'$t (dof='+ str(k) + ')$')  
    x = np.linspace(round(min(ts)), round(max(ts)+0.5), 200) 
    plt.plot(x,stats.t.pdf(x=x, df=k), label = r'$t (dof='+ str(k) + ')$')
    plt.legend()
    plt.show() 

def F(df1=10, df2=10, N = 1000):
    '''
    The F-distribution
    '''
    U = scipy.stats.chi2.rvs(df=df1, size=N)
    V = scipy.stats.chi2.rvs(df=df2, size=N)

    Fs = U/df1 / (V/df2)

    plt.figure(figsize = (10,3))
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.hist(Fs, bins=100, color = 'gray', edgecolor='black')
    plt.title("Frequency Histogram\ndegree of freedom = (" + str(df1) + ',' + str(df2) + '). simulations = ' + str(N))
    plt.show()

    plt.figure(figsize = (10,3)) 
    plt.title('Theoretical Distribution\n' + r'$F (dof1='+ str(df1) + ', dof2='+ str(df2) + ')$')  
    x = np.linspace(round(min(Fs)), round(max(Fs)+0.5), 200) 
    plt.plot(x,stats.f.pdf(x=x, dfn=df1, dfd=df2), label = r'$F (dof1='+ str(df1) + ', dof2='+ str(df2) + ')$')
    plt.legend()
    plt.show() 

def fibonacci(n):
    '''
    Get the fibonacci series

    Parameters
    ----------
    n : integer or array, e.g., 
        10 - return the 10-th item
        range(1,11) - return the top-10 items
    '''
    phi = (1 + 5**0.5)/2.0
    return np.round((np.power(phi, n) - np.power(1-phi, n)) / 5**0.5) # .astype(int) will soon exceed the int range

def benford(data="stock", N = 1000):
    '''
    Benford's law: also called the Newcomb–Benford law, the law of anomalous numbers, or the first-digit law, is an observation about the frequency distribution of leading digits in many real-life sets of numerical data. The law states that in many naturally occurring collections of numbers, the leading significant digit is likely to be small.
    本福特定律揭示了十进制数据的一个统计学规律，即首位数字出现的概率为：
    d 1 2 3 4 5 6 7 8 9
    p 30.1% 17.6% 12.5% 9.7% 7.9% 6.7% 5.8% 5.1% 4.6%

    Parameters
    ----------
    data : data set / experiment to use
        'stock' - use 20-year stock trading volume data of Apple Inc. (AAPL)
        'trade' - annual trade data for countries. https://comtrade.un.org/data/mbs
        'fibonacci' - use the top-N fibonacci series

    Note
    ----
    If for some reason, the AAPL.csv is missing, use the following code to retrieve:

        import yfinance as yf
        data = yf.download('AAPL','2000-01-01','2020-05-01') # may also try 'GOOG', etc.
        data.to_csv('AAPL.csv')
    '''
    volumes = []
    title = ''
    if data == 'stock':        
        volumes = pd.read_csv(DATA_FOLDER + '/AAPL.csv')['Volume'].values
        title = '20-year stock trading volume of AAPL'
    elif data == 'trade':
        volumes = pd.read_csv(DATA_FOLDER + '/MBSComtrade.csv')['value'].values
        title = 'Annual trade for countries (UN Comtrade Database)'
    elif data == 'fibonacci':
        volumes = fibonacci(range(1, N+1))
        title = 'Fibonacci series (Top-' + str(N) + ')'

    cnts = np.zeros(10)

    for v in volumes:        
        s = str(v).lstrip('0.') 
        if s and len(s) > 0:            
            leading_digit = int(s[0])        
            cnts[leading_digit] += 1

    total = np.sum(cnts)

    plt.figure(figsize = (10,3))
    plt.bar(range(1, len(cnts)), height = cnts[1:], alpha = 0.5)
    for a, b in zip(range(1, len(cnts)), cnts[1:]):
        plt.text(a, b, '%.0f %%' % (b*100/total), ha='center', va='bottom')

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.title("Frequency Histogram of Leading Digits\n"+ title)
    
    plt.figure(figsize = (10,3)) 
    plt.title('Theoretical Distribution\nBenford PMF')  
    percents = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
    percents = np.array(percents) / 100
    plt.bar(range(1, 10), height = percents, label = 'Benford PMF', alpha = 0.5)
    for a, b in zip(range(1, 10), percents):
        plt.text(a, b, '%.1f %%' % (b*100), ha='center', va='bottom')

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.legend()
    plt.show() 