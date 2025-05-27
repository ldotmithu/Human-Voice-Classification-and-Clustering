from src.Config.config_entity import DataValidationConfig
import os 
import pandas as pd 
from src.Utility.common import Create_Folder,Read_yaml
from pathlib import Path
path = "schema.yaml" 

class DataValidation:
    def __init__(self):
        self.validation = DataValidationConfig()
        self.schema = Read_yaml(path)['columns']
        
        Create_Folder(self.validation.root_dir)
    
    def check_columns(self):
        data = pd.read_csv(self.validation.data_path)
        all_columns = data.columns    
        req_columns = self.schema
        
        miss_columns = [col for col in req_columns if col not in all_columns]
        if miss_columns:
            Validation_Status =False
            with open(os.path.join(self.validation.root_dir,self.validation.status_path),'w') as f:
                    f.write(f"Validation_status : {Validation_Status}")
                    print(Validation_Status)
                    print(miss_columns)
        else:
            Validation_Status =True
            with open(os.path.join(self.validation.root_dir,self.validation.status_path),'w') as f:
                f.write(f"Validation_status : {Validation_Status}")
                print(Validation_Status)
                    
                
                
        