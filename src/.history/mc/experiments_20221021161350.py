import numpy as np
import random
import math
import matplotlib.pyplot as plt
from IPython.display import HTML, display

if __package__:
    from . import distributions
else:
    import distributions

def pi(N = 1000000, a = 4, l = 1,flavor = 1):
    
    """
    Perform a mc experiment to estimate PI. 
    
    Parameters
    ----------
    N : number of points.
    a : the distance between two parallel lines in the Buffon's needle problem.
    l : the length of the needle arbitrarily cast in the Buffon's needle problem.
    flavor : which implementation to use. 
        0 - the classic Buffon's needle problem.
        1, 2 - circle inside square. N points (x,y) are drawn from uniform random distributions in the range of -1 to +1. 
    The points within the unit circle divided by N is an approximation of PI/4.
    Returns
    -------
    freq : The ratio / percentage of points within the unit circle divided by N.
    PI :  An estimated value of PI.
    """
    if flavor == 0:
        xl = np.pi*np.random.random(N)
        yl = 0.5*a*np.random.random(N)
        m = 0
        for x,y in zip(xl,yl):
            if y < 0.5*l*np.sin(x):
                m += 1

        freq = m/N
        PI = 2*l/(a*freq)
        print("frequency = {}/{} = {}".format(m, N, m/N))
        print("PI = {}".format(2*l/(a*(m/N))))        

    elif flavor == 1:

        def is_inside_unit_circle(x, y):
            '''
            Determines whether the point (x,y) falls inside the unit circle.   
            '''
            return x**2 + y**2 < 1

        def unit_test():
            assert is_inside_unit_circle(0.5,-0.5) == True
            assert is_inside_unit_circle(0.999,0.2) == False
            assert is_inside_unit_circle(0,0) == True
            assert is_inside_unit_circle(0.5,-0.5) == True
            assert is_inside_unit_circle(-0.9,0.9) == False   
    
        xs = np.random.uniform(-1,1,N)
        ys = np.random.uniform(-1,1,N)
        # cnt 用来统计落在圆内点的数目
        cnt = 0
        for i in range(N):
            if (is_inside_unit_circle(xs[i], ys[i])):
                cnt += 1

        freq = cnt / N    
        PI = freq*4
        print("frequency = {}/{} = {}".format(cnt, N, cnt/N))
        print("PI = {}".format(cnt/N*4))       

    else:

        # Implementation 2: Monte-Carlo: PI
        pts = np.random.uniform(-1,1,(N,2))
        # Select the points according to your condition
        idx = (pts**2).sum(axis=1)  <= 1.0

        freq = idx.sum()/N
        PI = freq*4
        print("frequency = {}/{} = {}".format(idx.sum(), N, idx.sum()/N))
        print("PI = {}".format(idx.sum()/N*4))        
    return freq, PI
          

def parcel(N=100000, num_players = 5, num_ops = 10):
    
    """
    Simulate a bi-directional parcel passing game. 
    [num_players] players form a circle. 
    Then, each round the parcel can be passed to the left or right person. 
    球回到A手中的试验次数 / 总试验次数 = parcel(试验次数, 玩家数目，每次试验传球次数)
    Parameters
    ----------
    N : number of experiments.
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

    L=0
    history = []
    for iter in range(N):
        position = 0
        for op in range(num_ops):
            # random.choice([-1, +1]) # 随机产生-1或+1的函数，用于表示球左传(-1)或右传(+1)
            position = (position + random.choice([-1, +1]) + num_players) % num_players # %表示mod运算
        history.append(position)
        if(position == 0):
            L += 1

    return L/N


def dices(N = 10000):
    """
    Randomly roll three dice and calculate the probabilities of various situations. 
    The corresponding score for each dice point is as follows:
    If the three dice have 1, 2, 3 points or 4, 5 and 6 respectively, 16 points are awarded;
    If all three dice have the same number of points, 8 points are awarded;
    If only two of the dice have the same number of points, 2 points are awarded;
    If the three dice have different points, 0 points are awarded.
    Parameters
    ----------
    N : number of experiments.
    Returns
    -------
    Frequency History, i.e., the number of times each case occurs
    Notes
    -----
    When calculating the probability that the three dices have different points.
    it is necessary to remove the cases where the number of points is 1, 2, 3, and 4, 5, and 6, respectively.
    """

    samples = np.random.randint(low=1, high=7, size=(N,3)) # range: [low, high)
    dict_cnt = {}
    dict_cnt['ooo'] = 0 # 三个相同
    dict_cnt['123'] = 0
    dict_cnt['456'] = 0
    dict_cnt['xyz'] = 0 # 三个均不同，但需排除123和456的情况
    dict_cnt['oox'] = 0 # 两个相同

    for s in samples:
        if s[0] == s[1] and s[0] == s[2]:
            dict_cnt['ooo'] += 1 # 三个相同
        elif sorted(s) == [1,2,3]:
            dict_cnt['123'] += 1
        elif sorted(s) == [4,5,6]:
            dict_cnt['456'] += 1
        elif s[0] != s[1] and s[0] != s[2] and s[1] != s[2]:
            dict_cnt['xyz'] += 1 # 三个均不同，但需排除123和456的情况
        else:
            dict_cnt['oox'] += 1# 两个相同

    assert(dict_cnt['ooo'] + dict_cnt['xyz'] + dict_cnt['123'] + dict_cnt['456'] + dict_cnt['oox'] == N)

    for key in dict_cnt:
        dict_cnt[key] = dict_cnt[key] / N

    # print("Experiment Result", dict_cnt)

    #  Theoretical value：
    dict_tcnt = {}
    dict_tcnt['ooo'] = 6*(1/6)**3
    dict_tcnt['123'] = 6*(1/6)**3
    dict_tcnt['456'] = 6*(1/6)**3
    dict_tcnt['xyz'] = 6*5*4/(6**3)-dict_cnt['123']-dict_cnt['456']
    dict_tcnt['oox'] = 6*5*3/(6**3)

    html_str = '<h2>The dice experiment</h2><p>' + '''
    Randomly roll three dice and calculate the probabilities of various situations. 
    The corresponding score for each dice combination is as follows:
    (1) If the three dice have 1, 2, 3 points or 4, 5 and 6 respectively, 16 points are awarded;
    (2) If all three dice have the same number of points, 8 points are awarded;
    (3) If two of the dices have the same number of points, 2 points are awarded;
    (4) If the three dices have different points, 0 points are awarded.
    ''' + '</p>'
    html_str += '<table>'
    html_header = '<tr><th></th>'
    html_row1 = '<tr><td>Experimental Frequencies (f)<br/>N = ' + str(N) + '</td>'
    html_row2 = '<tr><td>Theoretical PMF (p)</td>'

    for key in dict_tcnt:
        html_header += '<th>'+ key + '</th>'
        dict_tcnt[key] = round( dict_tcnt[key] , 6)
        html_row1 += '<td>' + str(dict_cnt[key]) + '</td>'
        html_row2 += '<td>' + str(dict_tcnt[key]) + '</td>'

    # print("Theoretical PMF",dict_tcnt)

    html_str = html_str + html_header + '</tr>' + html_row1 + '</tr>' + html_row2 + '</tr>' + '</table>'
    display(HTML(html_str))
    
    return dict_cnt

def prisoners(n = 100, N = 2000):
    '''
    The famous 100-prisoners quiz.
    We will prove that the limit is (1-ln2) when n approaches +inf

    Parameters
    ----------
    n : how many prisoners
    N : MC experiments

    Returns
    -------
    frequency : of the N experiments, how many times the n prisoners survive.
    '''
    WINS = 0
    FAILS = 0
    for i in range(N): # i is MC experiment NO, we will not use it later.
        
        boxes_inner = np.random.choice(list(range(n)), n, replace = False)
        failed = False # at least one prisoner failed the test
        
        for j in range(n): # j is prisoner NO            
            found = False
            
            target_box_index = j
            for k in range(round(n/2)): # k is the draw round
                target_box_index = boxes_inner[target_box_index]
                if target_box_index == j:
                    found = True
                    break
            
            if found == False:
                # DOOMED
                failed = True
                break
        
        if failed:
            FAILS += 1
        else:
            WINS += 1
        
    return WINS / (WINS + FAILS)

def prisoners_limit(ns = [10, 20, 30, 40, 50, 100, 200, 500, 1000], N = 1000, repeat = 1, SD = 0):
    '''
    Test how the survival rate changes with n. The limit is 1-ln2.

    Parameters
    ----------
    ns : n values to be tested. default is [10, 20, 30, 50, 100, 200, 500, 1000, 2000]
    N : how many MC experiments to run for each n 
    repeat : repeat multiple times to calculate the SD (standard deviation)
    SD : how many SD (standard deviation) to show in the error bar chart 
    '''
    
    fss = []

    if repeat is None or repeat < 1:
        repeat = 1

    for _ in range(repeat):
        fs = []
        for n in ns:
            fs.append( prisoners(n, N = N) )
        fss.append(fs)
    
    fss = np.array(fss) # repeat-by-ns matrix
    fsm = fss.mean(axis = 0)
        
    plt.figure(figsize = (10, 4))

    if SD == 0 or repeat <= 1:
        plt.scatter(ns, fsm, label='survival chance')
        plt.plot(ns, fsm)
    else: # show +/- std errorbar
        plt.errorbar(ns, fsm, fss.std(axis = 0)*SD, 
                    # color = ["blue","red","green","orange"][c], 
                    linewidth=1, 
                    alpha=0.2,
                    label= 'survival chance. mean ± '+ str(SD) +' SD',
                    )
        plt.scatter(ns, fsm)
    
    plt.hlines(y = 1 - math.log(2), xmin = np.min(ns), xmax = np.max(ns), \
        color ='r', label = r'$1- ln(2) = $' + str(round(1 - math.log(2), 3))) 
    plt.xlabel('prisoner number')
    plt.legend()
    plt.show()

def galton_board(num_layers = 20, N = 5000, flavor=1, display = True):
    return distributions.binom(num_layers, N, flavor, display=display) 

def paper_clips(num_rounds = 10000, num_clips_k = 1.6, verbose = False):
    return distributions.zipf(num_rounds, num_clips_k, verbose)

def sudden_death(num_rounds = 1000, p = 0.01, N = 10000):
    return distributions.exponential(num_rounds, p, N)
