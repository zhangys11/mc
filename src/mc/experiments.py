import numpy as np
import random
from tqdm import tqdm

def pi(N = 1000000, flavor = 0):
    
    """
    Perform a mc experiment to estimate PI. 
    N points (x,y) are drawn from uniform random distributions in the range of -1 to +1. 
    The points within the unit circle divided by N is an approximation of PI/4.

    Parameters
    ----------
    N : number of points.
    flavor : which implementation to use. 1 or 2.

    Returns
    -------
    freq : The ratio / percentage of points within the unit circle divided by N.
    PI :  An estimated value of PI.
    """

    if flavor == 1:
        xs = np.random.uniform(-1,1,N)
        ys = np.random.uniform(-1,1,N)
        # cnt 用来统计落在圆内点的数目
        cnt = 0
        for i in range(N):
            if (is_inside_unit_circle(xs[i], ys[i])):
                cnt += 1

        freq = cnt / N        

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


def dices(N = 10000000):
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
   The number of times each case occurs and verifies that the final result satisfies the normalization.

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
    for s in tqdm(samples):
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
    sum=dict_cnt['ooo'] + dict_cnt['xyz'] + dict_cnt['123'] + dict_cnt['456'] + dict_cnt['oox']
    print(dict_cnt)
    print(dict_cnt['ooo']/sum + dict_cnt['xyz']/sum + dict_cnt['123']/sum + dict_cnt['456']/sum + dict_cnt['oox']/sum)# 满足归一化
    
    return 


def galton_board(m = 20, N = 5000, display = True):
    
    return binom() 


def paper_clips(num_rounds = 10000, num_clips_k = 1.6, verbose = False):
    
    return zipf()

