from cardinality_estimator import CardinalityEstimator
import math

class MinCount(CardinalityEstimator):
    """
    MinCount with stochastic averaging (Lumbroso, CORE algorithm).
    """

    def __init__(self, m=128, seed=None):
        self.m = m
        super().__init__(seed=seed)

    def reset(self):
        # Initialize registers to 1
        self.registers = [1.0] * self.m

    def add(self, element):
        A = self.hs(element)[0]/2**32  # A in (0,1) :(

        j = int(self.m * A)      # floor(mA)
        frac = self.m * A - j    # fractional part

        # j in [0, m-1]
        self.registers[j] = min(self.registers[j], frac)

    def estimate(self):
        Z = sum(self.registers)

        # avoid division by zero (degenerate case)
        if Z == 0:
            return float("inf")
        Z = self.m * (self.m - 1) / Z

        # small range correction (Linear Counting)
        V = self.registers.count(0)
        if Z <= 2.5 * self.m and V > 0:
            Z = self.m * math.log(self.m / V)

        return Z
    