# src/dashboard.py

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from data_preprocessing import preprocess_diabetes_data, preprocess_heart_data
from model_training import train_models

st.set_page_config(page_title="Healthcare Data Analysis", layout="wide")
st.title("🩺 Advanced Healthcare Data Analysis Dashboard")
st.markdown("""
This advanced dashboard allows you to:
- Predict **Diabetes** and **Heart Disease**
- Visualize **feature importance** of trained models
- Explore **dataset summary & correlations**
- Explain model predictions using **SHAP**
""")

# -------------------- Safe data paths --------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIABETES_PATH = os.path.join(PROJECT_DIR, "data", "diabetes.csv")
HEART_PATH = os.path.join(PROJECT_DIR, "data", "heart.csv")

# -------------------- Sidebar --------------------
dataset_choice = st.sidebar.selectbox("Select Dataset", ["Diabetes", "Heart Disease"])
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])


# -------------------- Functions --------------------
def load_data(dataset):
    if dataset == "Diabetes":
        return preprocess_diabetes_data(DIABETES_PATH)
    else:
        return preprocess_heart_data(HEART_PATH)


# -------------------- Main --------------------
X, y = load_data(dataset_choice)
models = train_models(X, y, dataset_name=dataset_choice)
model = models[model_choice]

st.subheader(f"{dataset_choice} Prediction & Insights")

# -------------------- Dataset Summary --------------------
with st.expander("📊 Dataset Summary"):
    st.write(X.describe())
with st.expander("🧩 Feature Correlation Heatmap"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

# -------------------- User Input --------------------
if dataset_choice == "Diabetes":
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose", 0, 300, 120)
    bloodpressure = st.number_input("BloodPressure", 0, 200, 70)
    skinthickness = st.number_input("SkinThickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 79)
    bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
    dpf = st.number_input("DiabetesPedigreeFunction", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 1, 120, 30)

    input_data = pd.DataFrame([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]],
                              columns=X.columns)

    if st.button("Predict Diabetes"):
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
        st.success(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
        st.info(f"Probability: {proba:.2f}")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(8, 5))
            sns.barplot(x=model.feature_importances_, y=X.columns)
            plt.title(f"{model_choice} Feature Importance")
            st.pyplot(plt)

        # SHAP explain
        if model_choice in ["Random Forest", "XGBoost"]:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(input_data)
            st.subheader("🔍 SHAP Explanation")
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(plt.gcf())

else:  # Heart Disease
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 0, 250, 130)
    chol = st.number_input("Cholesterol", 0, 600, 250)
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (0=No, 1=Yes)", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 0, 250, 150)
    exang = st.selectbox("Exercise Induced Angina (0=No,1=Yes)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal]],
                              columns=X.columns)

    if st.button("Predict Heart Disease"):
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
        st.success(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
        st.info(f"Probability: {proba:.2f}")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            sns.barplot(x=model.feature_importances_, y=X.columns)
            plt.title(f"{model_choice} Feature Importance")
            st.pyplot(plt)

        # SHAP explain
        if model_choice in ["Random Forest", "XGBoost"]:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(input_data)
            st.subheader("🔍 SHAP Explanation")
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(plt.gcf())
