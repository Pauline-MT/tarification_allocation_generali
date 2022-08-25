import numpy as np
import pandas as pd


class Sensitivity_Analysis:
    def __init__(self, data:pd.DataFrame, param_to_update:str):
        self.data = data
        self.param_to_update = param_to_update

    def bruit_gaussien(self, mean:float, standard_deviation:float):
        noise = np.random.normal(mean,standard_deviation,self.data.shape[0])
        colonne_a_modifier = self.data[self.param_to_update].to_numpy()
        colonne_noised = noise + colonne_a_modifier
        df = pd.DataFrame(colonne_noised, columns = ["noised"])
        return df

    def get_mean(self, data, param_to_update:str):
        mean = data[param_to_update].mean()
        return mean 