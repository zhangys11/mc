import numpy as np
from ..mcbase import McBase


class Pi(McBase):

    """
    Perform a mc experiment to estimate PI.

    Parameters
    ----------
    a : the distance between two parallel lines in the Buffon's needle problem.
    l : the length of the needle arbitrarily cast in the Buffon's needle problem.
    flavor : which implementation to use.
        0 - the classic Buffon's needle problem.
        1 - circle inside square. N points (x,y) are drawn from uniform random distributions in the range of -1 to +1.
        The points within the unit circle divided by N is an approximation of PI/4.

    Returns
    -------
    freq : The ratio / percentage of points within the unit circle divided by N.
    PI :  An estimated value of PI.
    """

    def __init__(self, N=1000000, a=4, l=1, flavor=1):
        super().__init__(None, N)
        self.a = a
        self.l = l
        self.flavor = flavor

    def run(self):
        if self.flavor == 0:
            xl = np.pi*np.random.random(self.N)
            yl = 0.5*self.a*np.random.random(self.N)
            m = 0
            for x, y in zip(xl, yl):
                if y < 0.5*self.l*np.sin(x):
                    m += 1

            freq = m/self.N
            PI = 2*self.l/(self.a*freq)
            print("frequency = {}/{} = {}".format(m, self.N, m/self.N))
            print("PI = {}".format(2*self.l/(self.a*(m/self.N))))

        elif self.flavor == 1:

            def is_inside_unit_circle(point_x, point_y):
                """
                Determines whether the point (x,y) falls inside the unit circle.
                """
                return point_x**2 + point_y**2 < 1

            def unit_test():
                assert is_inside_unit_circle(0.5, -0.5)
                assert is_inside_unit_circle(0.999, 0.2) is False
                assert is_inside_unit_circle(0, 0)
                assert is_inside_unit_circle(0.5, -0.5)
                assert is_inside_unit_circle(-0.9, 0.9) is False

            xs = np.random.uniform(-1, 1, self.N)
            ys = np.random.uniform(-1, 1, self.N)

            cnt = 0
            for i in range(self.N):
                if is_inside_unit_circle(xs[i], ys[i]):
                    cnt += 1

            freq = cnt / self.N
            PI = freq*4
            print("frequency = {}/{} = {}".format(cnt, self.N, cnt/self.N))
            print("PI = {}".format(cnt/self.N*4))

        else:
            # Implementation 2: Monte-Carlo: PI
            pts = np.random.uniform(-1, 1, (self.N, 2))
            # Select the points according to your condition
            idx = (pts**2).sum(axis=1) <= 1.0

            freq = idx.sum()/self.N
            PI = freq*4
            print("frequency = {}/{} = {}".format(idx.sum(), self.N, idx.sum()/self.N))
            print("PI = {}".format(idx.sum()/self.N*4))

        return freq, PI
