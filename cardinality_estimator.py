from abc import ABC, abstractmethod
import random
import randomhash

class CardinalityEstimator(ABC):
    """
    Abstract base class for cardinality estimators.
    """

    def __init__(self, seed=None, num_hashes=10):
        self.seed = seed
        random.seed(self.seed)
        rfh = randomhash.RandomHashFamily(count=num_hashes)
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

    def hs(self):
        """Compute the family of hashes"""
        pass
