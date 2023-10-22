from ..mcbase import McBase
from ..distributions import Zipf


class Paper_Clips(McBase):
    def __init__(self, N=10000, num_clips=16000):
        super().__init__(None, N)
        self.num_clips = num_clips

    def run(self, verbose=False, display=True):
        zipf = Zipf.Zipf(self.N, self.num_clips)
        zipf.run(verbose=verbose, display=display)
        return
