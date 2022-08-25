import time 
import numpy as np
import pandas as pd


from evaluator import Evaluator
from save_utils import save_all
from visualisor import plot_gains, plot_revaluations, plot_efficient_frontier, plot_average_revaluation

if __name__ == "__main__":

    dataset_name = "ptf_mrh_2020_elast_plr.csv"
    gamma_interval = np.linspace(0.85, 0.95, 10)
    debug = False 
    plot_each = False
    evaluate_sensitivity = False
    if debug: 
        gamma_interval = [0.90, 0.92]
    gains = {}
    predictions = {}
    for gamma in gamma_interval:
        params_1 = {"population_size": 30, "generations": 50, "gamma": gamma}
        params_2 = {"n_clusters": 50, "minimizer": "BFGS", "gamma": gamma}
        genetic_params = ("genetic_algorithm", params_1)
        clustered_bfgs_params = ("clustered_deterministic_minimizer", params_2)
        start = time.time()
        algorithm_list = [genetic_params, clustered_bfgs_params]
        gains[gamma], predictions[gamma] = Evaluator(dataset_name=dataset_name,
                                                     algorithm_list=algorithm_list,
                                                     gamma=gamma, 
                                                     evaluate_sensitivity=evaluate_sensitivity,
                                                     params_to_evaluate=["coeff_prix"],
                                                     debug=debug).evaluate()
        print(time.time() - start)
        if plot_each: 
            plot_gains(gains[gamma])
    save_all(predictions, gains)
    plot_efficient_frontier(gains)
    n = len(gamma_interval)
    plot_revaluations(predictions[gamma_interval[n//2]], gamma_interval[n//2])
    plot_average_revaluation(predictions, gains)
