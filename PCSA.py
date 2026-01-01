from cardinality_estimator import CardinalityEstimator
from utils import rho
import math

class PCSA(CardinalityEstimator):
    """
    Probabilistic Counting with Stochastic Averaging (PCSA).
    """
    def __init__(self, b=6, bitmap_size=32, seed=None):
        self.b = b
        self.m = 2**b
        self.bitmap_size = bitmap_size
        self.phi = 0.77351  # Correction factor
        self.red_size = self.bitmap_size-b
        super().__init__(seed)

    def reset(self):
        self.bitmaps = [[0] * self.bitmap_size for _ in range(self.m)]

    def add(self, element):
        y = self.hs(element)[0]
        alpha = y%self.m
        index = rho(y//self.m, size=self.red_size)-1
        self.bitmaps[alpha][index] = 1


    def estimate(self) -> float:
        S = 0
        for i in range(self.m):
            R = 0
            while self.bitmaps[i][R] and (R<self.bitmap_size):
                R+=1
            S += R

        return int((self.m/self.phi)*(2**(S/self.m)))
    

