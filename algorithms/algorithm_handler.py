from .genetic_algorithm import GeneticAlgorithm
from .deterministic_minimizer import DeterministicMinimizer, ClusteredDeterministicMinimizer

class AlgorithmHandler:
    def __init__(self, algorithm_name, data, **params):
        self.algorithm_name = algorithm_name
        self.data = data
        self.model = None
        self.train_algo(**params)

    def train_algo(self, **params):
        if "genetic_algorithm" in self.algorithm_name:
            ga = GeneticAlgorithm(data=self.data, **params)
            ga.train()
            self.model = ga 
        elif "clustered_deterministic_minimizer" in self.algorithm_name:
            clustered_optimizer = ClusteredDeterministicMinimizer(data=self.data, **params)
            clustered_optimizer.train()
            self.model = clustered_optimizer
        elif "deterministic_minimizer" in self.algorithm_name:
            optimizer = DeterministicMinimizer(data=self.data, **params)
            optimizer.train()
            self.model = optimizer

    def predict(self):
        return self.model.predict()