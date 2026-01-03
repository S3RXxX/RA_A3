import randomhash
from cardinality_estimator import CardinalityEstimator
import heapq

class REC(CardinalityEstimator):
    """
    Recordinality estimator.
    """
    def __init__(self, b=None, k=64, seed=None, do_hash=True):
        self.b = b
        if self.b is not None:
            self.k = 2**b
        else:
            self.k = k
        self.do_hash=do_hash
        super().__init__(seed=seed)

    def reset(self):
        # self.S = set()
        self.S = []      # min-heap of hash values
        self.R = 0

    def add(self, element):
        """Compute the k-records"""
        if self.do_hash:
            y = self.hs(element)[0]
        else:
            y = element

        # if element in self.S:
        #     return
        if y in self.S:
            return 
        
        # initial phase (fill S with the first k distinct elements)
        if len(self.S) < self.k:
            # self.S.add(element)
            heapq.heappush(self.S, y)
            self.R += 1
            return

        # S_min = min(self.S, key=lambda x: self.hs(x)[0])
        # if y > self.hs(S_min)[0]:
            # self.S.remove(S_min)
            # self.S.add(element)
        if y > self.S[0] and y not in self.S:
            heapq.heapreplace(self.S, y)
            self.R += 1
    
    def estimate(self):
        """Return Z"""
        return self.k*((1+(1/self.k))**(self.R-self.k+1))-1

    def __repr__(self):
        return f"REC(k={self.k}, do_hash={self.do_hash})"
