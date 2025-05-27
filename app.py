import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import os

# Paths
SCHEMA_PATH = "schema.yaml"
PREPROCESS_PATH = "artifacts/data_transfomation/preprocess.pkl"
MODEL_PATH = "artifacts/trainer/model.pkl"
METRICS_PATH = "artifacts/evaluation/metrics.json"

# Load schema
def load_schema(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Load metrics
def load_metrics(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

# Validate uploaded data
def validate_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns

# Preprocess data
def preprocess_data(df, preprocess_obj, top_features):
    df = df[top_features].dropna()
    return preprocess_obj.transform(df)

# Main Streamlit app
def main():
    st.title("Human Voice Gender Classification 🎙️")
    st.markdown("""
    Predict gender (Male/Female) from voice features using a Random Forest model. 📊
    Choose an input method: upload a CSV file or manually enter feature values.
    """)

    # Load schema and artifacts
    try:
        schema = load_schema(SCHEMA_PATH)
        preprocess_obj = joblib.load(PREPROCESS_PATH)
        model = joblib.load(MODEL_PATH)
        metrics = load_metrics(METRICS_PATH)
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return

    top_features = schema.get("top_features", [])[:-1]  # Exclude 'label'
    required_columns = schema.get("columns", [])

    # Display metrics
    if metrics:
        st.subheader("Model Performance Metrics 📈")
        st.write(f"Accuracy: {metrics.get('acc', 'N/A'):.4f}")
        st.write(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        st.write(f"MAE: {metrics.get('mae', 'N/A'):.4f}")
        st.write(f"R²: {metrics.get('r2', 'N/A'):.4f}")
    else:
        st.warning("Metrics file not found.")

    # Input method selection
    input_method = st.radio("Select Input Method", ("Upload CSV File 📄", "Manual Feature Entry ✍️"))

    if input_method == "Upload CSV File 📄":
        st.subheader("CSV File Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.dataframe(df.head())

                # Validate columns
                missing_columns = validate_columns(df, required_columns)
                if missing_columns:
                    st.error(f"Missing columns: {', '.join(missing_columns)}")
                    return

                # Preprocess and predict
                X_processed = preprocess_data(df, preprocess_obj, top_features)
                predictions = model.predict(X_processed)
                df['Predicted_Gender'] = ['Male' if pred == 1 else 'Female' for pred in predictions]

                # Display predictions
                st.subheader("Predictions")
                st.dataframe(df[['Predicted_Gender'] + top_features])

                # Pie chart
                st.subheader("Gender Prediction Distribution 📊")
                gender_counts = pd.Series(predictions).value_counts()
                labels = ['Male' if x == 1 else 'Female' for x in gender_counts.index]
                sizes = gender_counts.values
                colors = ['#66b3ff', '#ff9999']
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures pie is circular
                plt.title("Distribution of Predicted Genders")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing file: {e}")

    else:  # Manual Feature Entry
        st.subheader("Manual Feature Entry")
        st.write("Enter values for the following features (based on `schema.yaml`).")

        # Create form for manual input
        with st.form("manual_input_form"):
            feature_values = {}
            for feature in top_features:
                feature_values[feature] = st.number_input(
                    feature, 
                    min_value=float(-1000), 
                    max_value=float(1000), 
                    value=0.0, 
                    step=0.01,
                    format="%.4f"
                )
            submitted = st.form_submit_button("Predict Gender")

            if submitted:
                try:
                    # Create DataFrame from input
                    input_df = pd.DataFrame([feature_values], columns=top_features)
                    st.write("Input Features:")
                    st.dataframe(input_df)

                    # Preprocess and predict
                    X_processed = preprocess_data(input_df, preprocess_obj, top_features)
                    prediction = model.predict(X_processed)[0]
                    gender = 'Male' if prediction == 1 else 'Female'

                    # Display prediction
                    st.subheader("Prediction")
                    st.success(f"Predicted Gender: **{gender}**")

                except Exception as e:
                    st.error(f"Error processing input: {e}")

if __name__ == "__main__":
    main()