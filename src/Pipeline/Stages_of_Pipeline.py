from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transfomation import DataTransform
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation



from pathlib import Path

class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        ingestion = DataIngestion()
        ingestion.download_zipfile()
        ingestion.unzip_operation()
        
class DataValidationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        validation = DataValidation()
        validation.check_columns()
        
class DataTransformPipeline:
    def __init__(self):
        pass
    def main(self):
        transform = DataTransform()
        transform.initiate_preprocess()    
        
class ModelTrainPipeline:
    def __init__(self):
        pass
    def main(self):
        trainer = ModelTrainer()
        trainer.training()         
        
class ModelEvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        evaluation =  ModelEvaluation()
        evaluation.model_evaluation()           
        
            