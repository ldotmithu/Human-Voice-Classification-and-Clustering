from src.Config.config_entity import ModelTrainingConfig
import os 
import joblib
import numpy as np
from src.Utility.common import Create_Folder,Read_yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

parems_path = "parems.yaml"
class ModelTrainer:
    def __init__(self):
        self.trainer = ModelTrainingConfig()
        self.parems = Read_yaml(parems_path)['parameter']
        Create_Folder(self.trainer.root_dir)
        
    def training(self):
        train_data = np.load(self.trainer.train_data)
        train_X = train_data[:,:-1]
        train_y = train_data[:,-1] 
        
        
        rf = RandomForestClassifier(n_estimators = self.parems.get("n_estimators"),
                           random_state = self.parems.get("random_state"))
        rf.fit(train_X,train_y)
        print(rf.score(train_X,train_y))
        joblib.dump(rf,os.path.join(self.trainer.root_dir,self.trainer.model_path))
        print("save the model.pkl")