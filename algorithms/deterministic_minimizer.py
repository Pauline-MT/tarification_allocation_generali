import random as rd
import scipy.optimize
import pandas as pd

from .clustering import Clusterer
from .gain import average_retention, loss


class DeterministicMinimizer:
    def __init__(self, data:pd.DataFrame, minimizer:str, gamma:float=0.95):
        self.data = data
        self.pcc = data['pcc'].to_numpy()
        self.coeff_non_prix = data['coeff_non_prix'].to_numpy()
        self.coeff_prix = data['coeff_prix'].to_numpy()
        self.prime_profit = data['prime_profit'].to_numpy()
        self.minimizer = minimizer        
        self.gamma = gamma        
        self.slsqp_loss = loss
        self.bfgs_loss = self.gamma_loss
        self.retention = average_retention
        self.x0 = self.define_x0()
        self.constraints = self.define_constraints()
        self.result = None

    def define_x0(self):
        return [rd.uniform(-0.01, 0.01)
                for _ in range(len(self.data))]

    def define_bounds(self):
        lower_bound = -0.1
        upper_bound = 0.1
        return scipy.optimize.Bounds(lb=lower_bound, ub=upper_bound)

    def gamma_loss(self, price): 
        pen = 0 
        if self.retention(price, self.coeff_non_prix, self.coeff_prix) < self.gamma:
            pen = 10**10*(self.gamma - average_retention(price, self.coeff_non_prix, self.coeff_prix))
        return loss(price, self.pcc, self.coeff_non_prix, self.coeff_prix, self.prime_profit) + pen

    def gamma_opt_function(self, price):
        return self.retention(price, self.coeff_non_prix, self.coeff_prix) - self.gamma

    def define_constraints(self):
        function = self.gamma_opt_function
        const_dict = ({"type": 'ineq', "fun": function})
        return const_dict

    def train(self):
        if "SLSQP" in self.minimizer:
            self.result = scipy.optimize.minimize(self.slsqp_loss,
                                              self.x0, args=(
                                                  self.pcc, self.coeff_non_prix, 
                                                  self.coeff_prix, self.prime_profit,), method="SLSQP",
                                              bounds=self.define_bounds(),
                                              constraints=self.define_constraints())
        elif "BFGS" in self.minimizer:
            self.result = scipy.optimize.minimize(self.bfgs_loss,
                                              self.x0, method="L-BFGS-B",
                                              bounds=self.define_bounds())                            

    def predict(self):
        return self.result.x


class ClusteredDeterministicMinimizer:
    def __init__(self, data:pd.DataFrame, minimizer:str, n_clusters:int=8, gamma:float=0.95):
        self.data = data
        self.minimizer = minimizer
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.clusterer = Clusterer(self.data, self.n_clusters)

    def train(self):
        clustered_data = self.clusterer.return_clusters()
        data_reconstructor = []
        m = len(clustered_data)
        k = 0
        if m // 10 != 0:
            k = m//10
        else:
            k = m
        for i in range(m):
            if i % k == 0 or i == m-1:
                print(str(i+1)+"/{}".format(m))
            data_cluster = clustered_data[i]
            optimizer = DeterministicMinimizer(data_cluster, self.minimizer, self.gamma)
            optimizer.train()
            prediction = optimizer.predict()
            data_cluster["prediction"] = prediction
            data_reconstructor.append(data_cluster)
        self.data = self.clusterer.reconstruct_data(data_reconstructor)

    def predict(self):
        return self.data["prediction"].to_numpy()
