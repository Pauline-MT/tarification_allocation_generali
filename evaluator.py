import json
import numpy as np  
import pandas as pd

from typing import List, Tuple, Dict

from algorithms.gain import gain, average_retention
from algorithms.preprocessor import Preprocessor 
from algorithms.algorithm_handler import AlgorithmHandler

class Evaluator: 
    def __init__(self, dataset_name:str, algorithm_list: List[Tuple[str, Dict[str, float]]], gamma:float, 
        evaluate_sensitivity:bool=False, params_to_evaluate:List[str]=None, debug:bool=False):
        self.dataset_name = dataset_name
        self.algorithm_list = algorithm_list
        self.gamma = gamma 
        self.evaluate_sensitivity = evaluate_sensitivity
        self.params_to_evaluate = params_to_evaluate        
        self.debug = debug 
        self.gains = {}
        self.predictions = {}
    
    def evaluate(self):
        #evaluates algorithms on a dataset for the gain function
        data = pd.read_csv('data/' + self.dataset_name)
        if self.debug:
            data = data[0:2000]
        preprocessor = Preprocessor()
        preproc_data = preprocessor.preprocess_dataset(data, drop_outliers=True)
        for algorithm_name, params in self.algorithm_list:
            algorithm = AlgorithmHandler(algorithm_name, preproc_data, **params) 
            prediction = algorithm.predict() 
            evaluation = gain(prediction, preproc_data['pcc'].to_numpy(), 
                                    preproc_data['coeff_non_prix'].to_numpy(),
                                    preproc_data['coeff_prix'].to_numpy(), 
                                    preproc_data['prime_profit'].to_numpy())
            self.check_gamma_condition(prediction, preproc_data)
            self.gains[algorithm_name] = evaluation 
            self.predictions[algorithm_name] = prediction
            if self.evaluate_sensitivity: 
                for param_to_evaluate in self.params_to_evaluate: 
                    preproc_data = preprocessor.preprocess_dataset(data, drop_outliers=True, 
                                                    gaussian_noise = [param_to_evaluate, True, 0, 0.1])
                    algorithm = AlgorithmHandler(algorithm_name, preproc_data, **params) 
                    prediction = algorithm.predict() 
                    evaluation = gain(prediction, preproc_data['pcc'].to_numpy(), 
                                            preproc_data['coeff_non_prix'].to_numpy(),
                                            preproc_data['coeff_prix'].to_numpy(), 
                                            preproc_data['prime_profit'].to_numpy())
                    self.check_gamma_condition(prediction, preproc_data)
                    algorithm_name = algorithm_name + "_" + param_to_evaluate
                    self.gains[algorithm_name] = evaluation 
                    self.predictions[algorithm_name] = prediction
        return self.gains, self.predictions
    
    def check_gamma_condition(self, prediction, preproc_data):
        coeff_non_prix = preproc_data['coeff_non_prix'].to_numpy()
        coeff_prix = preproc_data['coeff_prix'].to_numpy()
        print(average_retention(prediction, coeff_non_prix, coeff_prix) > self.gamma)