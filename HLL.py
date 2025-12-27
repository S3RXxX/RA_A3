from cardinality_estimator import CardinalityEstimator
from utils import rho, J0
import math

class HLL(CardinalityEstimator):
    """
    HyperLogLog (HLL).
    """
    def __init__(self, b=6, bitmap_size=32, seed=None):
        self.b = b
        self.m = 2**b
        self.alpha_m = self._alpha(self.m)
        # print("Alpha_m obtained")

        self.bitmap_size = bitmap_size
        self.red_size = self.bitmap_size-math.ceil(math.log(self.m, 2))
        super().__init__(seed)

    def reset(self):
        self.registers = [0] * self.m

    def _rho(self, w) -> int:
        """
        Position of the leftmost 1-bit in w.
        """
        return rho(w, size=self.red_size)

    def add(self, element):
        y = self.hs(element)[0]
        j = y%self.m
        w = y//self.m
        rho_w = self._rho(w=w)

        if rho_w > self.registers[j]:
            self.registers[j] = rho_w

    def estimate(self) -> float:
        # indicator function (inversed)
        Z_inv = sum(2.0 ** (-r) for r in self.registers)

        
        E = self.alpha_m * (self.m ** 2) / Z_inv

        # small range correction (Linear Counting)
        V = self.registers.count(0)
        if E <= 2.5 * self.m and V > 0:
            E = self.m * math.log(self.m / V)
        # large range correction
        if E>(1/30)*2**32:
            E = -2**32*math.log(1-E/2**32)

        return E
    
    @staticmethod
    def _alpha(m: int) -> float:
        """
        Bias correction constant αₘ.
        """

        return 1/(m*J0(m))
        
        # constant approximation
        # return 0.72134

        # variable approximation
        # if m == 16:
        #     return 0.673
        # elif m == 32:
        #     return 0.697
        # elif m == 64:
        #     return 0.709
        # else:
        #     return 0.7213 / (1 + 1.079 / m)
    
if __name__=="__main__":
    seed = 373
    hll = HLL(seed=seed)
    hll.compute("a")
    # print(hll._rho(1))
