# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def preprocess_diabetes_data(filepath):
    """
    Preprocess Diabetes dataset
    """
    # Load CSV
    df = pd.read_csv(filepath)

    # Separate features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Numerical columns
    num_cols = X.columns.tolist()

    # Pipeline for numerical features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    X_processed = num_pipeline.fit_transform(X[num_cols])

    # Convert back to DataFrame
    X_processed = pd.DataFrame(X_processed, columns=num_cols)

    return X_processed, y


def preprocess_heart_data(filepath):
    """
    Preprocess Heart Disease dataset
    """
    # Load CSV
    df = pd.read_csv(filepath)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Categorical and numerical columns
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']
    num_cols = [col for col in X.columns if col not in cat_cols]

    # Pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    # Column names after one-hot encoding
    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
    all_features = num_cols + cat_features.tolist()

    X_processed = pd.DataFrame(X_processed, columns=all_features)

    return X_processed, y


if __name__ == "__main__":
    # Test preprocessing
    X_diabetes, y_diabetes = preprocess_diabetes_data("../data/diabetes.csv")
    print("Diabetes data shape:", X_diabetes.shape)

    X_heart, y_heart = preprocess_heart_data("../data/heart.csv")
    print("Heart data shape:", X_heart.shape)
