from abc import ABC, abstractmethod
import randomhash
from utils import datasets_path, datasets

class CardinalityEstimator(ABC):
    """
    Abstract base class for cardinality estimators.
    """

    def __init__(self, seed=None, num_hashes=1):
        self.seed = seed
        self.rfh = randomhash.RandomHashFamily(count=num_hashes, seed=self.seed)
        self.reset()

    @abstractmethod
    def reset(self):
        """Reset internal state."""
        pass

    @abstractmethod
    def add(self, element):
        """Process one element from the stream."""
        pass

    @abstractmethod
    def estimate(self) -> float:
        """Return the cardinality estimate."""
        pass

    def hs(self, element):
        """Compute the family of hashes"""
        return self.rfh.hashes(element)
    

    def compute(self, file):

        # with open(datasets_path+file+".txt") as f:
        #     for word in f.read().splitlines():
        #         self.add(word)
        # return self.estimate()

        for i in range(len(datasets)):  # temporal to test
            self.reset()
            with open(datasets_path+datasets[i]+".txt") as f:
                for word in f.read().splitlines():
                    self.add(word)
            print(f"{datasets[i]} {self.estimate()}")


