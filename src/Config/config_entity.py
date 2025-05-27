from dataclasses import dataclass
import os 
from pathlib import Path


@dataclass
class DataIngestionConfig:
    root_dir :Path = "artifacts/data_ingestion"
    URL:str="https://github.com/ldotmithu/Dataset/raw/refs/heads/main/human%20voice%20clustering.zip"
    local_data_path:Path = "artifacts/data_ingestion/data.zip"
    unzip_dir :Path = "artifacts/data_ingestion"
    
@dataclass
class DataValidationConfig:
    root_dir:Path = "artifacts/data_validation"
    data_path:Path = "artifacts/data_ingestion/vocal_gender_features_new.csv"
    status_path:Path = "status.txt"    
    
@dataclass 
class DataTransformConfig:
    root_dir :Path = "artifacts/data_transfomation"
    data_path:Path = "artifacts/data_ingestion/vocal_gender_features_new.csv"
    status_path :Path = "artifacts/data_validation/status.txt"
    preprocess_path :Path = "artifacts/data_transfomation/preprocess.pkl"    
    
@dataclass 
class ModelTrainingConfig:
    root_dir:Path = "artifacts/trainer"
    preprocess_path :Path = "artifacts/data_transfomation/preprocess.pkl"
    train_data :Path = "artifacts/data_transfomation/train.npy"
    model_path:Path = "model.pkl"     
    
    
@dataclass 
class ModelEvaluationConfig:
    root_dir:Path = "artifacts/evaluation"        
    model_path:Path = "artifacts/trainer/model.pkl"
    metrics_path:Path = "metrics.json"
    test_data:Path = "artifacts/data_transfomation/test.npy"    