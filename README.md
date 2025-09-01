# 🩺 Healthcare Data Analysis & Disease Prediction Dashboard

## Project Overview
This advanced dashboard allows you to predict **Diabetes** and **Heart Disease** using machine learning models. It also provides **interactive visualizations**, **feature importance analysis**, and **explainable AI** insights to help understand model predictions.

---

## Features
- Predict **Diabetes** and **Heart Disease** based on patient input
- Dataset summary & statistics
- Correlation heatmap of features
- Feature importance visualization (Random Forest / XGBoost)
- SHAP explainable AI plots for model interpretability
- Interactive dashboard using **Streamlit**

---

## Dataset
- **Diabetes dataset**: 768 samples, 8 features  
  (`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`)  
- **Heart Disease dataset**: 303 samples, 13 features  
  (`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`)  
- Stored in the `data/` folder  

---

## Machine Learning Models
- Logistic Regression  
- Random Forest  
- XGBoost  

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC

---

Healthcare_Data_Analysis/
├── data/
│   ├── heart.csv
│   └── diabetes.csv
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── dashboard.py
├── README.md      
├── requirements.txt  


