# Sampling distributions used in various hypothesis tests
#
# In statistics, a sampling distribution or finite-sample distribution is the probability 
# distribution of a given random-sample-based statistic.

import numpy as np
from numpy import mat
import collections
from scipy.stats import binom, chi2, f
from scipy.special import gamma
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing

if __package__:
    from . import experiments
else:
    import experiments

def clt(dist = 'bernoulli', sample_size = [1,2,5,20], N = 10000, display = True):
    """
    Central Limit Theorem

    For a population, given an arbitrary distribution.
    Each time from these populations randomly draw [sample_size] samples 
    (where [sample_size] takes the value in the [dist]), A total of [N] times. 
    The [N] sets of samples are then averaged separately. 
    The distribution of these means should be close to normal dist when [sample_size] is big enough.

    Parameters
    ----------
    dist : base / undeyling /atom distribution. 底层/原子分布
        'uniform' - a uniform distribution U(-1,1) is used.
        'expon' - an exponential distribution Expon(1) is used. 
        'poisson' - poisson distribution PI(1) is used. 
        'coin' / 'bernoulli' - {0:0.5,1:0.5}
        'tampered_coin' - {0:0.2,1:0.8} # head more likely than tail 
        'dice' - {1:1/6,2:1/6,3:1/6,4:1/6,5:1/6,6:1/6} 
        'tampered_dice' - {1:0.1,2:0.1,3:0.1,4:0.1,5:0.1,6:0.5} # 6 is more likely
        None - use 0-1 distribution {0:0.5,1:0.5} by default        
    sample_size : sample size to be averaged over / summed up.
        Can be an array / list, user can check how the histogram changes with sample size. 
    N : Number of experiments.
    """ 
    
    rows = len(sample_size)
    fig = plt.figure(figsize=(12, rows*3))
    plt.axis('off')
    
    if dist == 'uniform':
        
        f = lambda x : np.random.uniform(-1,1,x).mean()
        dist_name = "$U(-1,1)$"
        
    elif dist == 'expon' or dist == 'exponential':
                
        f = lambda x : np.random.exponential(scale = 1, size = x).mean()
        dist_name = "$Expon(1)$"
        
    elif dist == 'poisson':
                
        f = lambda x : np.random.poisson(lam = 1, size = x).mean()
        dist_name = "$\pi(1)$"    
        
    elif dist == 'dice':
        
        f = lambda x : np.random.choice(list(range(1,7)), x).mean()
        dist_name = "PMF {1:1/6,2:1/6,3:1/6,4:1/6,5:1/6,6:1/6}, i.e., dice"
        
    elif dist == 'tampered_dice':
        
        f = lambda x : np.random.choice(list(range(1,6)) + [6]*5, x).mean()
        dist_name = "PMF {1:0.1,2:0.1,3:0.1,4:0.1,5:0.1,6:0.5}, i.e., a tampered dice"
        
    elif dist == 'tampered_coin':
        
        f = lambda x : np.random.choice([0]+[1]*4, x).mean()
        dist_name = "PMF {0:0.2,1:0.8}, i.e., a tampered coin"

    else: # dist == 'coin' or 'bernoulli':
        
        f = lambda x : np.random.choice([0,1], x).mean()
        dist_name = "$Bernoulli(0.5), i.e., coin$"
        
        
    title = "Use " + dist_name +" to verify CLT (central limit theorem)"
    plt.title(title)

    for row_index, n in enumerate(sample_size):

        xbars = []
        for i in range(N): # MC试验次数
            xbar = f(n) # np.random.uniform(-1,1,n).mean() #
            xbars.append(xbar)   

        ax = fig.add_subplot(rows, 1, row_index + 1)
        # ax.axis('off')
        ax.hist(xbars, density=False, bins=100, facecolor="none", edgecolor = "black", \
                 label='sample size = ' + str(n))
        ax.legend()
        ax.set_yticks([])
            
    
    # plt.yticks([])
    plt.show()
    
    
def clt_all():
    '''
    Very the CLT (Central Limit Theorem) with all supported underlying dists
    '''
    for dist in ['uniform', 'expon', 'poisson', 'coin', 'tampered_coin', 'dice', 'tampered_dice']:
        print('-----------', dist, '-----------')
        clt(dist, sample_size = [1,2,5,20,50], N = 10000)


def chisq_gof_stat(dist = 'binom', K = 8, sample_size = 100, N = 10000):
    '''
    Verify the chisq statistic used in Pearson's Chi-Square Goodness-of-Fit Test. 
    验证皮尔逊卡方拟合优度检验的卡方分布假设

    Parameters
    ----------
    dist : what kind of population dist to use. Default we use binom, i.e., the Galton board
        'binom' / 'galton' - the population is binom
        'dice' - 6 * 1/6
    K : classes in the PMF
    N : how many MC experiments to run
    '''

    # test with b(n,p)

    chisqs = []

    for i in range(N): # MC试验次数
        
        if dist == 'binom' or dist =='galton':

            h = experiments.galton_board(K - 1, sample_size, display = False) # rounds, layers
            # print('experiment', h)

            chisq = 0

            for j in range(K):
                pj = binom.pmf(j,K-1,0.5)
                npj = sample_size * pj # theoretical
                fj = h[j]
                
                chisq = chisq + (fj - npj)**2 / npj
        
        elif dist == 'dice':

            h = collections.Counter( np.random.randint(0, 6, sample_size) )
            
            chisq = 0
            
            for j in range(6):
                pj = 1.0/6
                npj = sample_size * pj
                fj = h[j]
                # print(pj, npj, fj)
                
                chisq = chisq + (fj - npj)**2 / npj
            
            chisqs.append(chisq)

        chisqs.append(chisq)

    plt.figure()
    plt.hist(chisqs, density=False, bins=100, facecolor="none", edgecolor = "black")
    plt.title("Histogram of the GOF test statistic ($\chi^2 = \sum_{i=1}^{k}\dfrac{(f_{j}-np_{j})^2}{np_{j}}$)\n. \
        Population is " + dist + ", sample size="+str(sample_size)) # $b("+ str(K) +", 1/2)$
    plt.show()

    plt.figure()
    x = np.linspace(0, np.max(chisqs) ,100)
    plt.plot(x, chi2.pdf(x, df = K-1), lw=3, alpha=0.6, label='dof = ' + str(K-1), c = "black")
    plt.title('Theoretical Distribution\n$\chi^2(dof='+ str(K-1) + ')$') 
    plt.legend()
    plt.show()


def anova_stat(K = 10, n = 10, N = 10000):
    '''
    验证 ANOVA的F分布假设
    F = MSTR/MSE ~ F(k-1, n-k)

    The H0 assumes mu1=mu2=...=muK. 
    In this experiment, all samples are drawn from N(0,1)

    Parameters
    ----------
    K : classes / groups
    n : samples per group. Total sample count is [K]*[n]
    N : how many MC experiments to run
    '''

    FS = []

    for i in range(N): # MC试验次数
        
        X = np.random.normal (0, 1, size=(n,K))
        SSTR = n*((X.mean(axis = 0)-X.mean())**2).sum()
        MSTR = SSTR/(K-1)
        SSE = ((X - X.mean())**2).sum()
        MSE = SSE/(K*n-K) # 此处K*n为公式中n，样本总量
        
        F = 1.0*MSTR/MSE    
        FS.append(F)

    plt.hist(FS, density=False, bins=100, facecolor="none", edgecolor = "black")
    plt.title("Histogram of the ANOVA test statistic ($F = \dfrac{MSTR}{MSE}$)\n. \
        Population is N(0,1). " + str(K) + " groups, " + str(n) + " samples per group.")
    plt.show()

    plt.figure()
    x=np.linspace(0, np.max(FS), 100)
    plt.plot(x,f.pdf(x,dfn=K-1,dfd=n*K-K), lw=3, alpha=0.6, c = "black", \
        label = '$F(' + str(K-1) + ',' + str(n*K-K) + ')$')
    plt.title('Theoretical Distribution\n$F(' + str(K-1) + ',' + str(n*K-K) + ')$') 
    plt.legend()
    plt.show()

def kw_stat(dist = 'uniform', K = 3, n = 100, N = 10000):
    '''
    Verify the Kruskal-Wallis test statistic (H) is a X2 random variable.

    The Mann-Whitney or Wilcoxon test compares two groups while the Kruskal-Wallis test compares 3.
    Kruskal-Wallis test is a non-parametric version of one-way ANOVA. It is rank based.
    Kruskal-Wallis H: a X2 test statistic.

    Parameters
    ----------
    dist : population assumption. As KW test is non-parametric, the choice of dist doesn't matter.
        By default, we use unform.
    K : groups / classes
    n : samples per class. In this experiment, we use equal group size, i.e., n1=n2=n3=...
    N : how many MC experiments to run
    '''
    
    ni = n
    nT = K * ni

    Hs = []

    for i in tqdm(range(10000)): # MC试验次数
        
        if dist == 'uniform':
            y1 = np.random.uniform(0,1,ni) 
            y2 = np.random.uniform(0,1,ni)
            y3 = np.random.uniform(0,1,ni)
        else: # 'gaussian'
            y1 = np.random.randn(ni) # normal
            y2 = np.random.randn(ni)
            y3 = np.random.randn(ni)

        yall = y1.tolist() + y2.tolist() + y3.tolist()
        sorted_id = sorted(range(len(yall)), key = lambda k: yall[k])

        R1 = np.sum(sorted_id[:ni])
        R2 = np.sum(sorted_id[ni:ni+ni])
        R3 = np.sum(sorted_id[ni+ni:])

        H = 12/nT/(nT+1) * (R1**2 + R2**2 + R3**2) / ni - 3 * (nT + 1)
        
        Hs.append(H)

    plt.hist(Hs, density=False, bins=100, facecolor="none", edgecolor = "black")
    plt.title("Histogram of Kruskal-Wallis test's H statistic ($H = [{\dfrac{12}{n_{T}(n_{T}+1)}\sum_{i=1}^{k}\dfrac{R_{i}^2}{n_{i}}]-3(n_{T}+1)}$)\n. \
        Population is " + ("U(0,1). " if dist=='uniform' else "N(0,1). ") + str(K) + " groups, " + str(n) + " samples per group.")
    plt.show()

    x=np.linspace(np.min(Hs) - np.min(Hs), np.max(Hs) - np.min(Hs), 100) # 差一个平移，research later
    plt.figure()
    plt.plot(x, chi2.pdf(x, df = K-1), label='dof = ' + str(K - 1))
    plt.title('Theoretical Distribution\n$\chi^2(dof='+ str(K-1) + ')$') 
    plt.legend()
    plt.show()

def median_stat(k = 6, ni = 1000, n = 10000):
    '''
    This test is performed by analyzing multiple sets of independent samples.                                  
    Examine whether there is a significant difference in the median of the population from which they come.
    ----------
    ni : samples per class. In this experiment, all group sizes are equal. 
    k : groups / classes
    n : how many MC experiments to run
    '''
    N=ni*k 
    MTs = []
    for i in tqdm(range(n)):
        X = np.random.randint(0,100,[k,ni])

        Os = []
        for j in range(0,k):
            x_median = np.median(X[j])
            O_1i = 0
            for y in range(0,ni):
                if X[j][y] > x_median:
                    O_1i += 1
            Os.append(O_1i)

        X_median = np.median(X)
        a =0
        for j in range(0,k):
            for y in range(0,ni):
                if X[j][y] > X_median:
                    a += 1
                    
        accu =0
        for x in range(0,k):
            accu += ((Os[x]-(ni*a)/N)**2)/ni
        MT = (N**2/(a*(N-a)))*accu
        MTs.append(MT)

    plt.hist(MTs, density=False, bins=100, facecolor="none", edgecolor = "black")
    plt.title("Histogram of Median Test $MT$ statistic ($MT = \dfrac{N^2}{ab}\sum_{i=1}^{k}\dfrac{(O_{1i}-n_{i}a/N)^2}{n_{i}}$)")
    plt.show()

def fk_stat(n = 10, k = 5, N = 1000):
    '''
    The Fligner-Killeen test is a non-parametric test for homogeneity of group variances based on ranks.
    Verify the Fligner-Killeen Test statistic (FK) is a X2 random variable.

    Parameters
    ----------
    n : samples per class. In this experiment, all group sizes are equal. 
    K : groups / classes
    N : how many MC experiments to run
    '''
    FKs = []
    for i in tqdm(range(N)):
        X = np.random.randint(0,100,[k,n])
        X_normal = preprocessing.scale(X)
        a_j_bar = (X_normal.sum(axis=1))/n
        a_bar = X_normal.sum()/(n*k)
        sum = 0
        for j in range(0,k):
            sum = sum+k*(a_j_bar[j]-a_bar)**2
        FK = sum/X_normal.var()
        FKs.append(FK)
    plt.hist(FKs, density=False, bins=100, facecolor="none", edgecolor = "black")
    plt.title("Histogram of Fligner-Killeen test $FK$ statistic ($FK = \dfrac{\sum_{j=1}^{k}n_{j}(\overline{a_{j}}-\overline{a})^2}{s^2}$)")
    plt.show()

    x = np.linspace(np.min(FKs), np.max(FKs), 100)
    plt.figure()
    plt.plot(x, chi2.pdf(x, df = k-1), label='dof = ' + str(k - 1))
    plt.title('Theoretical Distribution\n$\chi^2(dof='+ str(k-1) + ')$') 
    plt.legend()
    plt.show() 

def levene_hov_stat(ni = 5, k = 2, n = 1000):
    '''
    Levene's test is used to test if k samples have equal variances. 
    Verify the Levene's Test statistic (W) is a X2 random variable.

    Parameters
    ----------
    ni : samples per class. In this experiment, all group sizes are equal. 
    K : groups / classes
    n : how many MC experiments to run
    '''
    N=ni*k 
    Ws = []
    for i in tqdm(range(n)):
        X = np.random.randn(k,ni)
        X_mat = mat(X)
        X_bar = X_mat.mean(axis = 1)
        Z_ij = X_mat-X_bar
        Zi_bar = Z_ij.mean(axis = 1)
        Z_i2 = [(i - np.mean(Z_ij))**2 for i in Zi_bar]
        W = ((N-k)/(k-1))*(sum([ni*i for i in Z_i2])/np.sum(np.square(Z_ij-Zi_bar)))
        Ws.append(W[0,0])

    plt.hist(Ws, density=False, bins=100, facecolor="none", edgecolor = "black")
    plt.title("Histogram of Hotelling's $W$ statistic ($W=\dfrac{N-k}{k-1}*\dfrac{\sum_{i=1}^{k}n_{i}(\overline{Z_{i.}}-\overline{Z_{..}})^2}{\sum_{i=1}^{k}\sum_{j=1}^{n_{i}}(Z_{ij}-\overline{Z_{i.}})^2}$)")
    plt.show()


    x=np.linspace(np.min(Ws), np.max(Ws), 100)
    plt.figure()
    plt.plot(x, chi2.pdf(x, df = k-1), label='dof = ' + str(k - 1))
    plt.title('Theoretical Distribution\n$\chi^2(dof='+ str(k-1) + ')$') 
    plt.legend()
    plt.show()


def bartlett_hov_stat(k = 5, ni = 10, n = 1000):
    '''
    Bartlett's test is used to test homoscedasticity, that is, if multiple samples are from populations with equal variances. 
    Verify the Bartlett's Test statistic is a X2 random variable.

    Parameters
    ----------
    k : groups / classes
    ni : samples per class. In this experiment, all group sizes are equal. 
    n : how many MC experiments to run
    '''
    N=ni*k 
    BTs = []
    for i in tqdm(range(n)):
        X = np.random.randn(k,ni)
        Si_2 = np.var(X, axis=1) 
        SP_2 = (1/(N-k))*sum([i * (ni-1) for i in Si_2])
        ln_Si2 = np.log([i for i in Si_2])
        BT = ((N-k)*np.log(SP_2)-sum([i * (ni-1) for i in ln_Si2]))/(1+(1/(3*(k-1)))*(k*((1/ni)-1/(N-k))))
        BTs.append(BT)

    plt.hist(BTs, density=False, bins=100, facecolor="none", edgecolor = "black")
    plt.title("Histogram of Hotelling's $\chi^2$ statistic ($\chi^2 = \dfrac{(N-k)\ln^{(S_{P}^2)}-\sum_{i=1}^{k}(n_{i}-1)\ln^{(S_{i}^2)}}{1+\dfrac{1}{3(k-1)}(\sum_{i=1}^{k}(\dfrac{1}{n_{i}})-\dfrac{1}{N-k})}$)")
    plt.show()


    x=np.linspace(np.min(BTs), np.max(BTs), 100)
    plt.figure()
    plt.plot(x, chi2.pdf(x, df = k-1), label='dof = ' + str(k - 1))
    plt.title('Theoretical Distribution\n$\chi^2(dof='+ str(k-1) + ')$') 
    plt.legend()
    plt.show()


def bartlett_sphericity_stat():
    '''
    For sign test, if H0 is true (m = m0), the N- and N+ both follow b(n,1/2)

    Parameters
    ----------
    dist : population assumption. As sign test is non-parametric, the choice of dist doesn't matter.
        By default, we use exponential. It's theoretical median is m = $\theta ln(2)$
    n : sample size.
    N : how many MC experiments to run
    '''



def sign_test_stat(dist = 'expon', n = 100, N = 10000):
    '''
    For sign test, if H0 is true (m = m0), the N- and N+ both follow b(n,1/2)

    Parameters
    ----------
    dist : population assumption. As sign test is non-parametric, the choice of dist doesn't matter.
        By default, we use exponential. It's theoretical median is m = $\theta ln(2)$
    n : sample size.
    N : how many MC experiments to run
    '''
    
    poss = []
    negs = []

    for i in tqdm(range(N)): # MC试验次数
        
        x = np.random.exponential(scale = 1, size = n)
        n_pos = len(np.where(x - np.log(2) > 0)[0])
        n_neg = len(np.where(x - np.log(2) < 0)[0])
        
        poss.append(n_pos)
        negs.append(n_neg)

    plt.hist(poss, density=True, bins=100, facecolor="none", edgecolor = "black")
    plt.title("Histogram of sign test's N+ statistic\n. \
        Population is expon(1). " + str(n) + " samples.")
    plt.show()

    plt.hist(negs, density=True, bins=100, facecolor="none", edgecolor = "black")
    plt.title("Histogram of sign test's N- statistic\n. \
        Population is expon(1). " + str(n) + " samples.")
    plt.show()

    plt.figure()
    plt.title ("b ({}, 0.5)".format(n))
    x = np.linspace(0,n,n+1)
    pmf = binom.pmf(x, n, 0.5)
    lb = round( min( np.min(poss), np.min(negs)) )
    ub = round( max( np.max(poss), np.max(negs)) )
    plt.plot(x[lb:ub],pmf[lb:ub])
    plt.title('Theoretical Distribution\n$b(n='+ str(n) + ',p=1/2)$')     
    plt.show()

def cochrane_q_stat(p = 0.5, K = 3, n = 100, N = 10000):
    '''
    Cochrane-Q test T statistic is a X2 (CHISQ) random variable.  
    Cochrane-Q is an extension of McNemar that support more than 2 samples/groups.  
    H0: the percentage of "success / pass" for all groups are equal. 

    Parameters
    ----------
    p : we draw from a Bernoulli population with p. p is the "success / pass" probability.
    K : groups / classes
    n : samples per class. In this experiment, all group sizes are equal, as Cochrane-Q is paired / dependent. 
    N : how many MC experiments to run
    '''
    
    Ts = []

    for i in tqdm(range(N)): # MC试验次数
        
        X = np.random.binomial(1, p, (n, K)) # return a nxK matrix of {0,1}
        T = (K-1)*(K*np.sum(X.sum(axis = 0)**2)-np.sum(X.sum(axis = 0))**2) / np.sum ((K - X.sum(axis = 1) ) * X.sum(axis = 1) )
        Ts.append(T) #  / (K * n) 

    plt.hist(Ts, density=False, bins=100, facecolor="none", edgecolor = "black")
    plt.title("Histogram of Cochrane-Q test's T statistic ($T = \dfrac{(k-1)[k\sum_{j=1}^{k}X_{.j}^2-(\sum_{j=1}^{k} X_{.j})^2]}{k\sum_{i=1}^{b}X_{i.}-\sum_{i=1}^{b} X_{i.}^2}$)\n. \
        Population is " + "Bernoulli(" + str(p) + "). " + str(K) + " groups, " + str(n) + " samples per group.")
    plt.show()

    x=np.linspace(np.min(Ts), np.max(Ts), 100)
    plt.figure()
    plt.plot(x, chi2.pdf(x, df = K-1), label='dof = ' + str(K - 1))
    plt.title('Theoretical Distribution\n$\chi^2(dof='+ str(K-1) + ')$') 
    plt.legend()
    plt.show()

def hotelling_t2_stat(n = 50, k = 2, N = 1000):
    '''
    The Hotelling T2- distribution was proposed by H. Hotelling for testing equality of means of two normal populations. 
    This functions verify the T2 statistic constructed from two multivariate Gussian follows the Hotelling's T2 distribution. 

    For k=1 the Hotelling T2- distribution reduces to the Student distribution, 
    and for any k>0 it can be regarded as a multivariate generalization of the 
    Student distribution
    Parameters
    ----------
    n : samples per class.  
    K : groups / classes.
    N : how many MC experiments to run.
    '''
  
    T2s = []
    for i in tqdm(range(N)):
        X = np.random.randn(k,n) # Draw from a standard normal dist. The returned X is de-meaned, no need to do (X-mu) afterwards.
        X_mat=mat(X)
        X1 = (X_mat.sum(axis=1))/n #x ba
        sum_xs = 0
        for j in range(0,n):
            sum_xs = sum_xs+(X_mat[:,j]-X1)*((X_mat[:,j]-X1).T)
        SIGMA = sum_xs/(n-1)
        T2 = (n*X1.T)*(np.linalg.inv(SIGMA))*X1
        T2s.append(T2[0,0])
    plt.hist(T2s, density=False, bins=100, facecolor="none", edgecolor = "black")
    plt.title("Histogram of Hotelling's $T^2$ statistic ($T^2 = n(\overline{X}-\mu)^{T}S^{-1}(\overline{x}-\mu)$)")
    plt.show()

    x = np.linspace(np.min(T2s), np.max(T2s), 100)
    y = ((gamma((n+1)/2))*((1+x/n)**(-(n+1)/2)))/((gamma((n-1)/2))*(gamma(1)*n))
    plt.figure()
    plt.plot(x,y, lw=3, alpha=0.6, c = "black", \
            label = '$T^2(' + str(k) + ',' + str(n+k-1) + ')$')
    plt.title('Theoretical Distribution $T^2(' + str(k) + ',' + str(n+k-1) + ')$ \n $p(x) = \dfrac{\Gamma((n+1)/2)x^{k/2-1}(1+x/n)^{-(n+1)/2}}{\Gamma((n-k+1)/2)\Gamma(k/2)n^{k/2}}$') 
    plt.legend()
    plt.show()