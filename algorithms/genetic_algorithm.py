import numpy as np

from geneticalgorithm2 import geneticalgorithm2 as ga

from .gain import loss, average_retention


class GeneticAlgorithm:
    def __init__(self, data, population_size, generations, gamma=0.95):
        self.data = data
        self.pcc = data['pcc'].to_numpy()
        self.coeff_non_prix = data['coeff_non_prix'].to_numpy()
        self.coeff_prix = data['coeff_prix'].to_numpy()
        self.prime_profit = data['prime_profit'].to_numpy()
        self.f = self.get_algorithm_function
        self.varbounds = self.get_bounds()
        self.gamma = gamma
        self.params = {
            'max_num_iteration': generations,
            'population_size': population_size,
            'mutation_probability': 0.85,
            'crossover_probability': 0.3,
        }
        self.model = ga(function=self.f, dimension=len(self.data), variable_type='real',
                        variable_boundaries=self.varbounds, function_timeout=10**6,
                        algorithm_parameters=self.params)

    def train(self):
        self.model.run(no_plot=True)

    def predict(self):
        print(self.f(self.model.output_dict['variable']))
        return self.model.output_dict['variable']

    def get_algorithm_function(self, price):
        pen = 0
        if average_retention(price, self.coeff_non_prix, self.coeff_prix) < self.gamma:
            pen = 10**10*(self.gamma - average_retention(price, self.coeff_non_prix, self.coeff_prix))
        return loss(price, self.pcc, self.coeff_non_prix, self.coeff_prix, self.prime_profit) + pen

    def get_bounds(self):
        return np.array([[-0.1, 0.1]]*len(self.data))
