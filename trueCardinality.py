from cardinality_estimator import CardinalityEstimator
import math

class TrueCardinality(CardinalityEstimator):
    """Calculate the true cardinality of the data stream"""

    def __init__(self):
        super().__init__()

    def reset(self):
        self.s = set()

    def add(self, element):
        self.s.add(element)

    def estimate(self) -> float:
        return len(self.s)
    
if __name__ == "__main__":
    ce = TrueCardinality()
    ce.compute("a")
    # print(ce.estimate())