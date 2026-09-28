"""
Trains and exports the final ensemble model artifacts for the Streamlit web application.
Saves the fitted ColumnTransformer, Logistic Regression model, CatBoost model, and configuration.
"""

import os
import json
import pickle
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from imblearn.over_sampling import BorderlineSMOTE

from src.data_loader import prepare_kohli_dataset, split_innings
from src.feature_engineering import prepare_engineered_dataset, MODEL_FEATURES
from src.preprocessor import get_feature_preprocessor


def train_and_export_models(csv_path: str = "IPL.csv", output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)
    print("Preparing dataset for export...")

    # Load data
    df = prepare_kohli_dataset(filepath=csv_path, batter="V Kohli", max_year=2023)
    dfi1, _ = split_innings(df)
    dfi1 = prepare_engineered_dataset(dfi1)

    X = dfi1.drop(columns="runs_total")
    y = dfi1["runs_total"]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # 1. Fit Preprocessor
    preprocessor = get_feature_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)

    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    with open(os.path.join(output_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Saved preprocessor to {output_dir}")

    # 2. Train and Save BorderlineSMOTE + Logistic Regression
    smote = BorderlineSMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train)

    log_model = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        max_iter=3000,
        random_state=42
    )
    log_model.fit(X_train_sm, y_train_sm)

    joblib.dump(log_model, os.path.join(output_dir, "logistic_model.joblib"))
    with open(os.path.join(output_dir, "logistic_model.pkl"), "wb") as f:
        pickle.dump(log_model, f)
    print(f"Saved Logistic model to {output_dir}")

    # 3. Train and Save CatBoost
    cat_features = [
        X_train.columns.get_loc("bat_pos"),
        X_train.columns.get_loc("bowler"),
        X_train.columns.get_loc("over_phase")
    ]

    cat_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=9,
        border_count=32,
        random_strength=5,
        class_weights=[1, 6],
        loss_function="Logloss",
        random_seed=42,
        verbose=0
    )
    cat_model.fit(X_train, y_train, cat_features=cat_features)

    cat_path = os.path.join(output_dir, "catboost_model.cbm")
    cat_model.save_model(cat_path)
    with open(os.path.join(output_dir, "catboost_model.pkl"), "wb") as f:
        pickle.dump(cat_model, f)
    print(f"Saved CatBoost model (.cbm and .pkl) to {output_dir}")

    # 4. Save Ensemble Config
    config = {
        "batter": "Virat Kohli",
        "logistic_weight": 0.30,
        "catboost_weight": 0.70,
        "decision_threshold": 0.550,
        "feature_columns": [col for col in MODEL_FEATURES if col != "runs_total"],
        "metrics": {
            "accuracy": 0.7651,
            "recall": 0.5217,
            "precision": 0.3158,
            "f1": 0.3934,
            "roc_auc": 0.6681,
            "pr_auc": 0.3030
        }
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Saved ensemble configuration to {config_path}")

    print("All production model artifacts exported successfully!")


if __name__ == "__main__":
    train_and_export_models()
