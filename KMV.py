import heapq
from cardinality_estimator import CardinalityEstimator

class KMV(CardinalityEstimator):
    """
    K-Minimum Values (KMV) cardinality estimator.
    """

    def __init__(self, b=None, k=64, seed=None):
        self.b = b
        if self.b is not None:
            self.k = 2**b
        else:
            self.k = k
        self.bitmap_size = 32
        super().__init__(seed=seed)

    def reset(self):
        self.S = []       # max-heap of k smallest hashes (store negatives)

    def add(self, element):
        y = self.hs(element)[0]

        if -y in self.S:
            return

        # fill phase
        if len(self.S) < self.k:
            heapq.heappush(self.S, -y)  # using negative cause its a min-heap
            return

        # keep k smallest values
        if y < -self.S[0]:
            heapq.heapreplace(self.S, -y)

    def estimate(self) -> float:
        if len(self.S) < self.k:
            return len(self.S)
        v_k = -self.S[0]/ (2**self.bitmap_size) # normalize hash to (0,1]
        return (self.k - 1) / v_k
        # return self.k / v_k
        
    def __repr__(self):
        return f"KMV(m={self.k})"
    
