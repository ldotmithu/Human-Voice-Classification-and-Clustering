import os 
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score
import numpy as np 
import json

def Create_Folder(path):
    try:
        os.makedirs(path,exist_ok=True)
        print(f"{path} Created")
    except Exception as e:
        raise e     
    
def Read_yaml(path):
    with open(path ,'r') as f:
        file = yaml.safe_load(f)
        print(f"{path} Read the yaml successfully")
        return file
    
def eval_metrics(actual, pred):
        acc = accuracy_score(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        
        return rmse, mae, r2 , acc
    
def save_json(path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)    