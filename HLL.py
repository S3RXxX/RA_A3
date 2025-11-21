import randomhash
from cardinality_estimator import CardinaltyEstimator

class HLL(CardinaltyEstimator):
    def __init__(self):
        super().__init__()