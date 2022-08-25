import pandas as pd
import numpy as np
import sys 
from pathlib import Path
sys.path.append(str(Path('../').absolute().parent))
from .sensitivity_analysis import Sensitivity_Analysis

class Preprocessor:
    def __init__(self):
        pass
    
    def drop_columns(self, data:pd.DataFrame):
        data = data.drop('proba_resil_0%', axis=1)
        data = data.drop('proba_resil_5%', axis=1)
        return data 

    def preprocess_dataset(self, data:pd.DataFrame, drop_outliers = False, quartile = 0.05, gaussian_noise = [None, True, 0, 0.1]):
        #249 NaN for prime_profit, 249 NaN for pcc
        data = data.dropna()
        data = self.drop_columns(data)

        param_to_update = gaussian_noise[0]
        pourcentage = gaussian_noise[1]
        moyenne = gaussian_noise[2]
        sigma = gaussian_noise[3]

        if drop_outliers:
            # Suppression des données aux élasticités prix négatives
            index = data[(data["coeff_prix"]<=0)].index
            data.drop(index, inplace = True)

            # Suppression des valeurs extrêmes
            Q1 = data.quantile(quartile)
            Q3 = data.quantile(1 - quartile)
            IQR = Q3 - Q1
            M = ((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).to_numpy()
            outlier_indexes = list(np.nonzero(np.any(M.astype(int) != 0, axis=1))[0])
            indexes_to_keep = set(range(data.shape[0])) - set(outlier_indexes)
            data = data.take(list(indexes_to_keep))
            
        # Ajout d'un bruit
        if param_to_update != None:
            sa = Sensitivity_Analysis(data, param_to_update)
            if pourcentage:
                mean = sa.get_mean(data, param_to_update)              
                data[param_to_update] = sa.bruit_gaussien(moyenne*mean, abs(sigma*mean))["noised"].values
            else :
                data[param_to_update] = sa.bruit_gaussien(moyenne, sigma)["noised"].values

            if drop_outliers:
                index = data[(data["coeff_prix"]<=0)].index
                data.drop(index, inplace = True)
                
        return data