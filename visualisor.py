import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from algorithms.preprocessor import Preprocessor

def plot_gains(gains: dict):
    algo_names = gains.keys()
    highest_gains = gains.values()
    data_to_plot = pd.DataFrame(
        {"algo_names": algo_names, "highest_gains": highest_gains})
    sns.catplot(x="algo_names", y="highest_gains",
                kind="bar", data=data_to_plot)
    plt.show()


def plot_revaluations(predictions, gamma):
    for algorithm in predictions.keys():
        plt.hist(predictions[algorithm], density=True, bins=30)
        plt.xlabel('Revaluations δ')
        plt.ylabel('Probability')
        plt.title('%s with gamma = %s' % (algorithm, round(gamma, 2)))
        plt.show()


def plot_efficient_frontier(gains):
    gains_for_gamma = {}
    gamma_list = []
    for algorithm in gains[list(gains.keys())[0]].keys():
        gains_for_gamma[algorithm] = []
    for gamma in gains.keys():
        gamma_list.append(gamma)
        for algorithm in gains[gamma].keys():
            gains_for_gamma[algorithm].append(gains[gamma][algorithm])

    for algorithm in gains_for_gamma.keys():
        gains_rounded_list = np.round(gains_for_gamma[algorithm], 2)
        gamma_float_list = [float(gamma) for gamma in gamma_list]
        gamma_rounded_list = np.round(np.array(gamma_float_list), 2)
        plt.plot(gamma_rounded_list, gains_rounded_list)
        plt.title('Frontière efficiente pour l´algorithme %s' % algorithm)        
        plt.xlabel('Taux de rétention γ')
        plt.ylabel('Gain')
        plt.show()



def plot_average_revaluation(predictions, gains):
    data = pd.read_csv('data/' + "ptf_mrh_2020_elast_plr.csv")
    preprocessor = Preprocessor()
    preproc_data = preprocessor.preprocess_dataset(data)

    average_revaluation = {}
    for algorithm in gains[list(gains.keys())[0]].keys():
        gamma_list = []
        for gamma in gains.keys():
            gamma_list.append(gamma)
        pcc = preproc_data['pcc']

        average_revaluation[algorithm] = []

        for gamma in gains.keys():
            nominator = 0
            denominator = 0
            pcc_list = pcc.values.tolist()

            for i in range(min(len(pcc_list), len(predictions[gamma][algorithm]))):
                nominator = nominator + pcc_list[i] * \
                    (1+predictions[gamma][algorithm][i])
                denominator = denominator + pcc_list[i]
            if denominator == 0:
                denominator = 1
            average_revaluation[algorithm].append(nominator/denominator)

        plt.plot(gamma_list, average_revaluation[algorithm])
        plt.title('Revalorisations moyennes pour l´algorithme %s' % algorithm)
        plt.xlabel('Taux de rétention γ')
        plt.ylabel('Revalorisation (1 + δ)')
        plt.show()
