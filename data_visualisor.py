import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

dataset_name = "ptf_mrh_2020_elast_plr.csv"
df = pd.read_csv('data/' + dataset_name)[0:2000]

def affichage_outliers(data, column):
    plt.boxplot(data[column])
    plt.show()

def affichage_histogram(data, column):
    if column == "coeff_non_prix":
        data.coeff_non_prix.hist()
    elif column == "coeff_prix":
        data.coeff_prix.hist()
    elif column == "prime_profit":
        data.prime_profit.hist()
    elif column == "pcc":
        data.prime_profit.hist() 
    plt.show()

def affichage_scatter(data, column):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.scatter(data['id_police'], data[column])
    ax.set_xlabel('Client_Id')
    ax.set_ylabel(column)
    plt.show()

def zscore(data, nombre_sigma):
    z = np.abs(stats.zscore(data['coeff_prix']))
    threshold = nombre_sigma
    return(np.where(z>threshold))