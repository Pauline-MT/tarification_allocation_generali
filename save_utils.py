import json 
import numpy as np

def save_all(predictions_dict, gains_dict):
    clean_dicts(predictions_dict)
    #save the predictions into json files, right now overwriting each time
    with open('predictions.json', 'w') as preds: 
        json.dump(predictions_dict, preds)
    with open('gains.json', 'w') as gains:
        json.dump(gains_dict, gains)
    
def clean_dicts(predictions_dict):
    for gamma in predictions_dict.keys():
        for algo_name in predictions_dict[gamma].keys():
            if type(predictions_dict[gamma][algo_name]) == np.ndarray:
                predictions_dict[gamma][algo_name] = predictions_dict[gamma][algo_name].tolist()