# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import preprocessing functions
from data_preprocessing import preprocess_diabetes_data, preprocess_heart_data


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    return accuracy, precision, recall, f1, roc_auc


def plot_feature_importance(model, feature_names, title):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
        plt.title(title)
        plt.show()
    else:
        print(f"{title} - Model has no feature_importances_ attribute")


def train_models(X, y, dataset_name="Dataset"):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\n{name} training on {dataset_name}...")
        model.fit(X_train, y_train)
        print(f"{name} evaluation on {dataset_name}:")
        evaluate_model(model, X_test, y_test)
        plot_feature_importance(model, X.columns, f"{dataset_name} - {name} Feature Importance")
        trained_models[name] = model

    return trained_models


if __name__ == "__main__":
    # Diabetes dataset
    X_diabetes, y_diabetes = preprocess_diabetes_data("../data/diabetes.csv")
    print("\n--- Training models on Diabetes Dataset ---")
    diabetes_models = train_models(X_diabetes, y_diabetes, dataset_name="Diabetes")

    # Heart dataset
    X_heart, y_heart = preprocess_heart_data("../data/heart.csv")
    print("\n--- Training models on Heart Disease Dataset ---")
    heart_models = train_models(X_heart, y_heart, dataset_name="Heart Disease")
