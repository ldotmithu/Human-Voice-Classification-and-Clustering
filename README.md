# Human Voice Classification and Clustering 🎙️✨

Welcome to a machine learning project that classifies and clusters human voices by gender using the `vocal_gender_features_new.csv` dataset! 🚀 This project features a robust pipeline for data ingestion, validation, transformation, model training, and evaluation, topped with an interactive Streamlit web app for gender predictions via CSV upload or manual feature entry. 🌟

## Features 🌈
- **Data Ingestion** 📥: Downloads and unzips the dataset from a GitHub URL.
- **Data Validation** ✅: Ensures required columns match `schema.yaml`.
- **Data Transformation** 🔄: Preprocesses numerical features with StandardScaler, saves train/test splits as `.npy` files.
- **Model Training** 🧠: Trains a Random Forest classifier with parameters from `parems.yaml`.
- **Model Evaluation** 📊: Measures performance with accuracy, RMSE, MAE, and R², saved to `metrics.json`.
- **Streamlit App** 💻:
  - 📄 Upload a CSV file to predict gender for multiple samples and view a pie chart of Male/Female predictions.
  - ✍️ Manually enter feature values for a single gender prediction.
  - 📈 Displays model metrics (accuracy, RMSE, MAE, R²).
- **Clustering** 🔗: Implements K-Means clustering (see `voice_classification.py` for more).

## Requirements 🛠️
Dependencies listed in `requirements.txt`:
- `streamlit` 🌐
- `pandas` 🐼
- `numpy` 🔢
- `geopy` 🌍
- `scikit-learn` 🤖
- `xgboost` ⚡
- `matplotlib` 📉
- `seaborn` 🎨
- `joblib` 💾
- `PyYAML` 📋

## Installation ⚙️
1. **Clone the Repository** 📂:
   ```bash
   git clone https://github.com/ldotmithu/Human-Voice-Classification-and-Clustering.git
   cd Human-Voice-Classification-and-Clustering
   ```

2. **Create and Activate a Virtual Environment** 🖥️:
   ```bash
   conda create -n ml_pro python=3.10 -y
   conda activate ml_pro 
  
   ```

3. **Install Dependencies** 📦:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Pipelines** 🏭:
   Execute the pipeline scripts to prepare data and train the model:
   ```bash
   python -m src.components.data_ingestion
   python -m src.components.data_validation
   python -m src.components.data_transfomation
   python -m src.components.model_trainer
   python -m src.components.model_evaluation
   ```

5. **Run the Streamlit App** 🌍:
   ```bash
   streamlit run app.py
   ```
   Open the URL (e.g., `http://localhost:8501`) to explore the app! 🎉

## Directory Structure 📁
```
Human-Voice-Classification-and-Clustering/
├── artifacts/                     # 🗄️ Pipeline outputs
│   ├── data_ingestion/            # 📥 Dataset
│   ├── data_validation/           # ✅ Validation status
│   ├── data_transfomation/        # 🔄 Preprocessed data
│   ├── trainer/                   # 🧠 Trained model
│   ├── evaluation/                # 📊 Metrics
├── src/                           # 🛠️ Source code
│   ├── Config/                    # ⚙️ Configurations
│   ├── Utility/                   # 🧰 Utilities
│   ├── components/                # 🏭 Pipeline scripts
├── data/                          # 📄 Local dataset (not tracked)
├── schema.yaml                    # 📋 Feature schema
├── parems.yaml                    # 🔧 Model parameters
├── app.py                         # 🌐 Streamlit app
├── voice_classification.py        # 🔗 Clustering script
├── requirements.txt               # 📦 Dependencies
├── .gitignore                     # 🚫 Ignored files
├── README.md                      # 📖 This file
```

## Usage 🎮
- **Pipelines** 🏭: Run the pipeline scripts in order to ingest, validate, transform, train, and evaluate.
- **Streamlit App** 💻:
  - **CSV Upload** 📄:
    - Upload a CSV with features from `schema.yaml` (e.g., `mfcc_5_mean`, `mean_spectral_contrast`).
    - View predicted genders, model metrics, and a pie chart showing Male/Female prediction distribution.
  - **Manual Feature Entry** ✍️:
    - Input values for 15 top features (e.g., `mfcc_5_mean`, `mfcc_3_std`).
    - Get a single gender prediction (Male/Female).

**Example**:
```bash
streamlit run app.py
```

## Dataset 📊
- **File**: `vocal_gender_features_new.csv`
- **Source**: [GitHub](https://github.com/ldotmithu/Human-Voice-Classification-and-Clustering.git) 📥
- **Features**: 43 acoustic features (e.g., `mean_spectral_centroid`, `mfcc_1_mean`) and `label` (0 = Female, 1 = Male).
- **Note**: Downloaded to `artifacts/data_ingestion/`.

## Evaluation Metrics 📈
- **Classification**: Accuracy (primary), RMSE, MAE, R² (for compatibility).
- Stored in `artifacts/evaluation/metrics.json`.

## License 📜
[MIT License](LICENSE) (to be added).

- **Dataset Issues** 📄: Ensure `vocal_gender_features_new.csv` is in `artifacts/data_ingestion/`.
- **Dependencies** 📦: Verify installation:
  ```bash
  pip install -r requirements.txt
  ```
- **Streamlit Errors** 💻: Check `preprocess.pkl`, `model.pkl`, `metrics.json`, and `schema.yaml` exist.

## Contact 📬
- **Author**: ldotmithu
- **Email**: ldotmithurshan222@gmail.com ✉️
- **GitHub**: [ldotmithu](https://github.com/ldotmithu) 🌐

Happy coding! 🎉