import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import base64
import io
import joblib

# Page Config
st.set_page_config(page_title="SimpleML for Teachers", layout="centered")

# Language Toggle
lang = st.sidebar.radio("Language / Bahasa", ["English", "Bahasa Malaysia"])

# Titles
if lang == "English":
    st.title("📘 SimpleML for Teachers")
    st.markdown("This app helps identify students at risk in Additional Mathematics.")
    upload_label = "Upload student data (.csv):"
    predict_label = "Predict At-Risk Students"
    download_label = "Download Results"
    output_label = "📊 Prediction Results"
else:
    st.title("📘 SimpleML untuk Guru")
    st.markdown("Aplikasi ini membantu mengenal pasti pelajar berisiko dalam Matematik Tambahan.")
    upload_label = "Muat naik data pelajar (.csv):"
    predict_label = "Ramalkan Pelajar Berisiko"
    download_label = "Muat turun Keputusan"
    output_label = "📊 Keputusan Ramalan"

# File upload
uploaded_file = st.file_uploader(upload_label, type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Data Preview")
    st.dataframe(df)

    # Encode categorical columns
    if "JANTINA" in df.columns:
        df["JANTINA"] = df["JANTINA"].map({"Perempuan": 0, "Lelaki": 1})

    if "GREDSPM" in df.columns:
        grade_map = {
            "A+": 1, "A": 2, "A-": 3, "B+": 4, "B": 5,
            "C+": 6, "C": 7, "D": 8, "E": 9, "G": 10
        }
        df["GREDSPM"] = df["GREDSPM"].map(grade_map)

    # Drop non-feature columns
    features = df.drop(columns=["NAMA", "Name", "At_Risk", "Risk_Level"], errors='ignore')

    # Check and drop any non-numeric columns that might still exist
    features = features.select_dtypes(include=["number", "bool"])

    # Load ML model
    model = joblib.load("simpleml_model.pkl")

    # Make predictions
    prediction = model.predict(features)
    df["Risk_Level"] = ["At Risk" if p == 1 else "Safe" for p in prediction]

    if st.button(predict_label):
        st.subheader(output_label)
        st.dataframe(df[["NAMA", "Risk_Level"]] if "NAMA" in df.columns else df)

        # SHAP Explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown("#### 🔍 Model Explainability (SHAP)")
        shap.summary_plot(shap_values, features, plot_type="bar", show=False)
        st.pyplot(bbox_inches='tight')

        # Download CSV
        csv = df.to_csv(index=False).encode()
        st.download_button(
            label=download_label,
            data=csv,
            file_name='simpleml_predictions.csv',
            mime='text/csv',
        )
