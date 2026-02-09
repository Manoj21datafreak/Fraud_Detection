import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_recall_curve
)


# -------------------------
# Data Loading
# -------------------------

def load_data(path):
    df = pd.read_csv(path)
    return df


# -------------------------
# Data Preparation
# -------------------------

def prepare_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def scale_features(X_train, X_test):

    scaler = StandardScaler()

    cols = ["Time", "Amount"]

    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])

    return X_train, X_test, scaler


# -------------------------
# Models
# -------------------------

def train_baseline_lr(X_train, y_train):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model


def train_balanced_lr(X_train, y_train):

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# -------------------------
# Evaluation
# -------------------------

def evaluate_model(name, model, X_test, y_test):

    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print(f"\n=== {name} ===")

    print("PR-AUC:",
          round(average_precision_score(y_test, y_scores), 4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return y_scores


def evaluate_thresholds(y_test, y_scores, thresholds):

    for th in thresholds:

        y_pred = (y_scores >= th).astype(int)

        print(f"\n--- Threshold {th} ---")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))


# -------------------------
# Main Pipeline
# -------------------------

def main():

    print("Loading data...")

    df = load_data("data/creditcard.csv")

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, scaler = scale_features(X_train, X_test)

    print("Train fraud ratio:", round(y_train.mean(), 5))
    print("Test fraud ratio:", round(y_test.mean(), 5))


    # Baseline LR
    baseline = train_baseline_lr(X_train, y_train)
    evaluate_model("Baseline Logistic Regression",
                   baseline, X_test, y_test)


    # Balanced LR
    balanced = train_balanced_lr(X_train, y_train)
    scores_bal = evaluate_model("Balanced Logistic Regression",
                                balanced, X_test, y_test)


    # Threshold tuning
    print("\n=== Threshold Tuning (Balanced LR) ===")

    thresholds = [0.9, 0.7, 0.5, 0.3, 0.1, 0.05]

    evaluate_thresholds(y_test, scores_bal, thresholds)


    # Random Forest
    rf = train_random_forest(X_train, y_train)
    scores_rf = evaluate_model("Random Forest",
                               rf, X_test, y_test)


    # Select best model (RF by default)
    best_model = rf
    best_scores = scores_rf

    OPTIMAL_THRESHOLD = 0.9

    print("\nSelected Model: Random Forest")
    print("Selected Threshold:", OPTIMAL_THRESHOLD)


    # Save artifacts
    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nModel and scaler saved.")


if __name__ == "__main__":
    main()
