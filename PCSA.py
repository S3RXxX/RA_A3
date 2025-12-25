from cardinality_estimator import CardinalityEstimator
import math

class PCSA(CardinalityEstimator):
    """
    Probabilistic Counting with Stochastic Averaging (PCSA).
    """
    def __init__(self, bitmap_size=64, seed=None):
        self.bitmap_size = bitmap_size
        self.phi = 0.77351  # Correction factor
        super().__init__(seed)

    def reset(self):
        self.bitmap = [0] * self.bitmap_size

    def _leading_zeros(self, x: int) -> int:
        """
        Number of leading zeros in a 64-bit integer.
        """
        if x == 0:
            return 64
        return 64 - x.bit_length()

    def add(self, element):
        y = self.hs(element)[0]
        p = self._leading_zeros(y)

        if p < self.bitmap_size:
            self.bitmap[p] = 1
        # print("bitmap", self.bitmap)

    def estimate(self) -> float:
        R = 0
        for bit in self.bitmap:
            if bit == 1:
                R += 1
            else:
                break

        return self.phi * (2 ** R)
    

if __name__=="__main__":  # just for testing, remove later
    pcsa = PCSA(seed=373)
    # pcsa.add("a")
    pcsa.compute("a")
    print(pcsa.bitmap)

