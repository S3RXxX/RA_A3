from cardinality_estimator import CardinalityEstimator
from utils import rho

class AdaptiveSampling(CardinalityEstimator):
    """
    Adaptive Sampling cardinality estimator.
    """

    def __init__(self, m=64, seed=None):
        self.m = m
        self.bitmap_size = 32
        super().__init__(seed=seed)

    def reset(self):
        self.S = {}   # element -> hash
        self.depth = 0

    def _leading_zeros(self, x):
        return rho(x, size=self.bitmap_size)-1

    def add(self, element):
        y = self.hs(element)[0]

        # check hash(z) = 0^p...
        if self._leading_zeros(y) < self.depth:
            return

        if element in self.S:
            return

        self.S[element] = y

        # memory overflow -> increase p and filter
        # if len(self.S) > self.m:
        # c = 0
        while len(self.S) > self.m:
            c+=1
            self.depth += 1

            # filter S
            self.S = {
                e: h for e, h in self.S.items()
                if self._leading_zeros(h) >= self.depth
            }
        # if c>1: print(c)

    def estimate(self):
        return len(self.S) * (2 ** self.depth)

if __name__ == "__main__":
    adaptiveSampling = AdaptiveSampling(m=256)
    adaptiveSampling.compute("a")