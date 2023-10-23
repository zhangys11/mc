import numpy as np
from IPython.display import HTML, display
from ..mcbase import McBase


class Dices(McBase):

    """
    Randomly roll three dice and calculate the probabilities of various situations.
    The corresponding score for each dice point is as follows:
    If the three dice have 1, 2, 3 points or 4, 5 and 6 respectively, 16 points are awarded;
    If all three dice have the same number of points, 8 points are awarded;
    If only two of the dice have the same number of points, 2 points are awarded;
    If the three dice have different points, 0 points are awarded.

    Returns
    -------
    Frequency History, i.e., the number of times each case occurs

    Notes
    -----
    When calculating the probability that the three dices have different points.
    it is necessary to remove the cases where the number of points is 1, 2, 3, and 4, 5, and 6, respectively.
    """

    def __init__(self, N=10000):
        super().__init__(None, N)

    def run(self):
        # range: [low, high)
        samples = np.random.randint(low=1, high=7, size=(self.N, 3))
        dict_cnt = {}
        dict_cnt['ooo'] = 0  # 三个相同
        dict_cnt['123'] = 0
        dict_cnt['456'] = 0
        dict_cnt['xyz'] = 0  # 三个均不同，但需排除123和456的情况
        dict_cnt['oox'] = 0  # 两个相同

        for s in samples:
            if s[0] == s[1] and s[0] == s[2]:
                dict_cnt['ooo'] += 1  # 三个相同
            elif sorted(s) == [1, 2, 3]:
                dict_cnt['123'] += 1
            elif sorted(s) == [4, 5, 6]:
                dict_cnt['456'] += 1
            elif s[0] != s[1] and s[0] != s[2] and s[1] != s[2]:
                dict_cnt['xyz'] += 1  # 三个均不同，但需排除123和456的情况
            else:
                dict_cnt['oox'] += 1  # 两个相同

        assert(dict_cnt['ooo'] + dict_cnt['xyz'] +
               dict_cnt['123'] + dict_cnt['456'] + dict_cnt['oox'] == self.N)

        for key in dict_cnt:
            dict_cnt[key] = dict_cnt[key] / self.N

        # print("Experiment Result", dict_cnt)

        #  Theoretical value：
        dict_tcnt = {}
        dict_tcnt['ooo'] = 6*(1/6)**3
        dict_tcnt['123'] = 6*(1/6)**3
        dict_tcnt['456'] = 6*(1/6)**3
        dict_tcnt['xyz'] = 6*5*4/(6**3)-dict_cnt['123']-dict_cnt['456']
        dict_tcnt['oox'] = 6*5*3/(6**3)

        html_str = '<h3>The dice experiment</h3><p>' + '''
        Randomly roll three dice and calculate the probabilities of various situations. 
        The corresponding score for each dice combination is as follows:
        <br/>(1) If the three dice have 1, 2, 3 points or 4, 5 and 6 respectively, 16 points are awarded;
        <br/>(2) If all three dice have the same number of points, 8 points are awarded;
        <br/>(3) If two of the dices have the same number of points, 2 points are awarded;
        <br/>(4) If the three dices have different points, 0 points are awarded.
        ''' + '</p>'
        html_str += '<table>'
        html_header = '<tr><th></th>'
        html_row1 = '<tr><td>Experimental Frequencies (f)<br/>N = ' + \
            str(self.N) + '</td>'
        html_row2 = '<tr><td>Theoretical PMF (p)</td>'

        for key in dict_tcnt:
            html_header += '<th>' + key + '</th>'
            dict_tcnt[key] = round(dict_tcnt[key], 5)
            html_row1 += '<td>' + str(dict_cnt[key]) + '</td>'
            html_row2 += '<td>' + str(dict_tcnt[key]) + '</td>'

        # print("Theoretical PMF",dict_tcnt)

        html_str = html_str + html_header + '</tr>' + \
            html_row1 + '</tr>' + html_row2 + '</tr>' + '</table>'
        display(HTML(html_str))

        return dict_cnt
