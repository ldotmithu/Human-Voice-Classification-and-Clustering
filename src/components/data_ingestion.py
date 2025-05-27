import os 
from src.Config.config_entity import DataIngestionConfig
from src.Utility.common import Create_Folder
from urllib.request import urlretrieve
import zipfile



class DataIngestion:
    def __init__(self):
        self.ingestion = DataIngestionConfig()
        Create_Folder(self.ingestion.root_dir)
    
    def download_zipfile(self):
        if not os.path.exists(path=self.ingestion.local_data_path):
            urlretrieve(self.ingestion.URL,self.ingestion.local_data_path)
            print("Zip file Downloaded")
        else:
            print(f"{self.ingestion.local_data_path} File Already Exists")    
    
    def unzip_operation(self):
        unzip_path = self.ingestion.unzip_dir
        with zipfile.ZipFile(self.ingestion.local_data_path) as f:
            f.extractall(unzip_path)
            print("Unzip Operation successfully completed")        
            
        
        
        