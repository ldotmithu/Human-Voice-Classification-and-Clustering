# Human Voice Gender Classification 🎙️

This project implements a machine learning pipeline to predict gender (Male/Female) from vocal features using a Random Forest Classifier. The pipeline includes data ingestion, validation, transformation, model training, evaluation, and a Streamlit web application for user interaction. 🛠️

## Project Overview 📖

The project processes a dataset of vocal features to classify gender. It downloads a ZIP file containing the dataset, validates it against a schema, preprocesses the data, trains a Random Forest model, evaluates its performance, and provides a Streamlit app for predictions via CSV upload or manual feature entry. The pipeline is modular, with separate components for each stage, and uses configuration files for flexibility. 📊

## Features ✨

- **Data Ingestion** 📥: Downloads and unzips a dataset from a specified URL.
- **Data Validation** ✅: Checks if the dataset contains required columns as defined in `schema.yaml`.
- **Data Transformation** 🔄: Preprocesses data using StandardScaler and saves train/test splits as `.npy` files.
- **Model Training** 🧠: Trains a Random Forest Classifier with parameters from `parems.yaml`.
- **Model Evaluation** 📈: Computes metrics (Accuracy, RMSE, MAE, R²) and saves them as JSON.
- **Streamlit App** 🌐: Provides a web interface for predictions, supporting CSV uploads and manual feature input, with visualization of prediction distributions.

## Requirements 🛠️

- Python 3.8+ 🐍
- Libraries listed in `requirements.txt` (e.g., pandas, scikit-learn, streamlit, numpy, joblib, pyyaml, matplotlib) 📚
- Internet connection for dataset download 🌐
- Git for cloning the repository 📂

## Installation 🔧

1. **Clone the Repository** 📥:
   ```bash
   git clone https://github.com/ldotmithu/Human-Voice-Classification-and-Clustering.git
   cd Human-Voice-Classification-and-Clustering
   ```

2. **Set Up a Virtual Environment** (recommended) 🖥️:
   ```bash
   conda activate -n ml-pro python=3.10 -y
   conda activate ml-pro
   ```

3. **Install Dependencies** 📦:
   ```bash
   pip install -r requirements.txt
   ```


## Project Structure 📂

```
human-voice-gender-classification/
├── artifacts/                    # Stores downloaded data, models, and metrics 📊
├── src/
│   ├── components/              # Pipeline components (ingestion, validation, etc.) 🧩
│   ├── Config/                 # Configuration classes ⚙️
│   ├── Pipeline/               # Pipeline orchestration 🔄
│   ├── Utility/                # Common utility functions 🛠️
├── app.py                       # Streamlit web application 🌐
├── main.py                      # Main script to run the pipeline 🚀
├── schema.yaml                  # Dataset schema configuration 📋
├── parems.yaml                  # Model parameters configuration ⚙️
├── requirements.txt             # Python dependencies 📦
├── setup.py                    # Project setup script 🔧
├── README.md                    # This file 📖
```

## Usage 🚀

### Running the Pipeline ⚙️

Execute the full pipeline (ingestion, validation, transformation, training, evaluation):
```bash
python main.py
```

This will:
1. Download and unzip the dataset to `artifacts/data_ingestion/` 📥.
2. Validate the dataset against `schema.yaml` ✅.
3. Preprocess the data and save train/test splits as `.npy` files 🔄.
4. Train a Random Forest model and save it as `model.pkl` 🧠.
5. Evaluate the model and save metrics to `metrics.json` 📈.

### Running the Streamlit App 🌐

Launch the Streamlit app for interactive predictions:
```bash
streamlit run app.py
```

The app provides two input methods:
- **CSV Upload** 📄: Upload a CSV file with vocal features to predict gender for multiple samples.
- **Manual Entry** ✍️: Input feature values manually to predict gender for a single sample.

The app displays model performance metrics and a pie chart of prediction distributions (for CSV uploads) 📊.

### Example Dataset 📊

The dataset is downloaded from:
```
https://github.com/ldotmithu/Dataset/raw/refs/heads/main/human%20voice%20clustering.zip
```

It contains vocal features (e.g., meanfreq, sd, median) and a target column (`label`) indicating gender (1 for Male, 0 for Female).

## Notes 📝

- Ensure the dataset URL is accessible and the ZIP file contains `vocal_gender_features_new.csv`.
- The Streamlit app requires `schema.yaml`, `preprocess.pkl`, `model.pkl`, and `metrics.json` to be present in the `artifacts/` directory.
- If you encounter issues with the dataset or paths, verify the configurations in `schema.yaml` and `parems.yaml`.
- The project assumes the dataset has no missing values or duplicates after preprocessing.


## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact 📧
Author: L.Mithurshan 
Project: Human Voice Gender Classification
For questions or issues, please open an issue on GitHub or contact [ldotmithurshan222@gmail.com].