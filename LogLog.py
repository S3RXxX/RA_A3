from HLL import HLL
import math

class LogLog(HLL):
    """
    LogLog
    """
    def __init__(self, b=6, bitmap_size=32, seed=None):
        super().__init__(b, bitmap_size, seed)

    
    def estimate(self) -> float:

        avg = sum(r-1 for r in self.registers)/self.m
        E = self.alpha_m * (self.m) * 2.0** avg
        
        return E
        # No correction?
        # small range correction (Linear Counting)
        # V = self.registers.count(0)
        # if E <= 2.5 * self.m and V > 0:
        #     E = self.m * math.log(self.m / V)
        # # large range correction
        # if E>(1/30)*2**32:
        #     E = -2**32*math.log(1-E/2**32)

        
    @staticmethod
    def _alpha(m: int) -> float:
        """
        Bias correction constant αₘ.
        """

        return (math.gamma(-1/m)*((2**(-1/m)-1)/math.log(2)))**(-m)
    