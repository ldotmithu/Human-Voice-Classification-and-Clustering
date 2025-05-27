from src.Config.config_entity import DataTransformConfig 
import os 
from src.Utility.common import Create_Folder,Read_yaml
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

schema_path = "schema.yaml" 

class DataTransform:
    def __init__(self):
        self.transform = DataTransformConfig()
        self.schema = Read_yaml(schema_path)
        Create_Folder(self.transform.root_dir)
    
    def preprocess_step(self):
        num_pipeline = Pipeline([
            ('num_pipeline',StandardScaler())
        ])    
        
        #power_pipeline = Pipeline([
            #("power_pipeline",PowerTransformer(method='yeo-johnson', standardize=False))
        #])
        preprocess = ColumnTransformer([
            ('num_pipeline',num_pipeline,self.schema['num-columns']),
            #("power_pipeline",power_pipeline,self.schema['power_columns'])
        ])
        
        return preprocess
    
    def initiate_preprocess(self):
        data = pd.read_csv(self.transform.data_path)
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        
        data = data[self.schema.get("top_features",[])]

        preprocess_obj = self.preprocess_step()
        
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        
        target_columns = self.schema.get('target',[])
        
        train_data_input_feature = train_data.drop(columns=target_columns, axis=1)
        train_data_target_feature = train_data[target_columns]
        
        test_data_input_feature = test_data.drop(columns = target_columns,axis = 1)
        test_data_target_feature = test_data[target_columns]
        
        train_pre = preprocess_obj.fit_transform(train_data_input_feature)
        test_pre = preprocess_obj.transform(test_data_input_feature)
        
        train_arr = np.c_[train_pre, np.array(train_data_target_feature)]
        test_arr = np.c_[test_pre, np.array(test_data_target_feature)]
        
        np.save(os.path.join(self.transform.root_dir,'train.npy'),train_arr)
        np.save(os.path.join(self.transform.root_dir,'test.npy'),test_arr)
        print("Train and test .npy files saved successfully.")
        
        joblib.dump(preprocess_obj,self.transform.preprocess_path)