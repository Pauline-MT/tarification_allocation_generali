import numpy as np
import pandas as pd 

from numba import jit

@jit(nopython=True, parallel=True)
def retention_probability(a: float, b: float, delta: float):
    return 1/(1+np.exp(a + 5*b*100*delta))

@jit(nopython=True, parallel=True)
def intermediate_gain(ac: float, a: float, b: float, delta: float, p: float):
    return (p*(1 + delta) - ac)*retention_probability(a, b, delta)

@jit(nopython=True, parallel=True)
def average_retention(price:np.array, coeff_non_prix:np.array, coeff_prix:np.array):
    retention = retention_probability(coeff_non_prix, coeff_prix, price)
    return np.mean(retention)

@jit(nopython=True, parallel=True)
def gain(price:np.array, pcc:np.array, coeff_non_prix:np.array, 
        coeff_prix:np.array, prime_profit:np.array):
    cost = intermediate_gain(pcc, coeff_non_prix, coeff_prix, price, prime_profit)
    return np.sum(cost)

@jit(nopython=True, parallel=True)
def loss(price:np.array, pcc:np.array, coeff_non_prix:np.array, 
        coeff_prix:np.array, prime_profit:np.array):
    return -1*gain(price, pcc, coeff_non_prix, coeff_prix, prime_profit)


def intermediate_data_retention(price:np.array, data:pd.DataFrame):
    retention = np.zeros(len(data))
    for i in range(len(data)):
        a = data["coeff_non_prix"].to_numpy()[i]
        b = data["coeff_prix"].to_numpy()[i]
        delta = price[i]
        retention[i] = retention_probability(a, b, delta)
    return retention