import os 
from pathlib import Path

main_folder = "src"

list_of_path = [
    f"{main_folder}/__init__.py",
    f"{main_folder}/components/__init__.py",
    f"{main_folder}/components/data_ingestion.py",
    f"{main_folder}/components/data_validation.py",
    f"{main_folder}/components/data_transfomation.py",
    f"{main_folder}/components/model_trainer.py",
    f"{main_folder}/components/model_evaluation.py",
    f"{main_folder}/Config/__init__.py",
    f"{main_folder}/Config/config_entity.py",
    f"{main_folder}/Pipeline/__init__.py",
    f"{main_folder}/Pipeline/Stages_of_Pipeline.py",
    f"{main_folder}/Utility/__init__.py",
    f"{main_folder}/Utility/common.py",
    "app.py",
    "main.py",
    "requirements.txt",
    "setup.py",
    "schema.yaml",
    "parems.yaml",
    
]

for path in list_of_path:
    file_path = Path(path)
    folder = file_path.parent
    os.makedirs(folder,exist_ok=True)
    
    if not os.path.exists(file_path):
        file_path.touch(exist_ok=True)