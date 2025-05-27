import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Data Preparation and Preprocessing
def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df = df.dropna()
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, X.columns

"""
# Exploratory Data Analysis
def perform_eda(df, output_dir='eda_plots'):
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Feature distribution
    plt.figure(figsize=(15, 10))
    df.hist(bins=30)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distribution.png')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()

"""    

# Model Development and Evaluation
def train_and_evaluate_models(X_train, X_val, y_train, y_val, feature_names):
    results = {}
    
    # Clustering: K-Means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train)
    kmeans_silhouette = silhouette_score(X_train, kmeans.labels_)
    results['KMeans_Silhouette'] = kmeans_silhouette
    
    # Clustering: DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X_train)
    if len(set(dbscan.labels_)) > 1:  # Check if clusters are formed
        dbscan_silhouette = silhouette_score(X_train, dbscan.labels_)
    else:
        dbscan_silhouette = -1  # Invalid silhouette score
    results['DBSCAN_Silhouette'] = dbscan_silhouette
    
    # Classification: Random Forest with Feature Selection and Hyperparameter Tuning
    rf = RandomForestClassifier(random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=10)
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_val_rfe = rfe.transform(X_val)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train_rfe, y_train)
    best_rf = grid_search.best_estimator_
    
    y_pred_rf = best_rf.predict(X_val_rfe)
    results['RandomForest'] = {
        'Accuracy': accuracy_score(y_val, y_pred_rf),
        'Precision': precision_score(y_val, y_pred_rf),
        'Recall': recall_score(y_val, y_pred_rf),
        'F1': f1_score(y_val, y_pred_rf)
    }
    
    # Classification: SVM
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_val)
    results['SVM'] = {
        'Accuracy': accuracy_score(y_val, y_pred_svm),
        'Precision': precision_score(y_val, y_pred_svm),
        'Recall': recall_score(y_val, y_pred_svm),
        'F1': f1_score(y_val, y_pred_svm)
    }
    
    # Classification: Neural Network
    nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_val)
    results['NeuralNetwork'] = {
        'Accuracy': accuracy_score(y_val, y_pred_nn),
        'Precision': precision_score(y_val, y_pred_nn),
        'Recall': recall_score(y_val, y_pred_nn),
        'F1': f1_score(y_val, y_pred_nn)
    }
    
    # Save confusion matrix for Random Forest
    cm = confusion_matrix(y_val, y_pred_rf)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.savefig('eda_plots/confusion_matrix_rf.png')
    plt.close()
    
    return results, best_rf, rfe, scaler

# Streamlit Application
def create_streamlit_app(best_rf, rfe, scaler, feature_names):
    st.title('Human Voice Classification')
    st.write('Upload a CSV file with voice features to predict gender.')
    
    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
    
    if uploaded_file is not None:
        # Read and preprocess uploaded data
        df = pd.read_csv(uploaded_file)
        if set(feature_names).issubset(df.columns):
            X = df[feature_names]
            X_scaled = scaler.transform(X)
            X_rfe = rfe.transform(X_scaled)
            
            # Predict
            predictions = best_rf.predict(X_rfe)
            df['Predicted_Gender'] = ['Male' if pred == 1 else 'Female' for pred in predictions]
            
            st.write('Predictions:')
            st.dataframe(df[['Predicted_Gender']])
        else:
            st.error('Uploaded CSV must contain the same feature columns as the training dataset.')

# Main Execution
if __name__ == '__main__':
    # Load and preprocess data
    file_path = 'notebook/data_set.csv'
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = load_and_preprocess_data(file_path)
    
    # Perform EDA
    df = pd.read_csv(file_path)
    # perform_eda(df.drop('label', axis=1))
    
    # Train and evaluate models
    results, best_rf, rfe, scaler = train_and_evaluate_models(X_train, X_val, y_train, y_val, feature_names)
    
    # Print results
    print("Model Evaluation Results:")
    for model, metrics in results.items():
        if isinstance(metrics, dict):
            print(f"\n{model}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            print(f"\n{model}: {metrics:.4f}")
    
    # Run Streamlit app
    create_streamlit_app(best_rf, rfe, scaler, feature_names)