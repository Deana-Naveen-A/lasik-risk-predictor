"""
LASIK Risk Prediction Project
================================================================
File    : src/train.py
Purpose : Train a separate XGBoost classifier for each of the
          three post-operative complications and evaluate
          performance on the held-out test set.

Models trained:
    1. Dry Eye Severity         (multiclass: 0, 1, 2)
    2. Night Vision Disturbance (binary: 0, 1)
    3. Ectasia Risk             (binary: 0, 1)

Each model is trained independently because the complications
have different class structures and clinical drivers. This
approach also allows each model to be updated or replaced
independently as new data becomes available.

Output:
    models/model_dry_eye.pkl
    models/model_night_vision.pkl
    models/model_ectasia.pkl
    outputs/evaluation/classification_reports.txt

Usage:
    py src/train.py
================================================================
"""

import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt

# Resolve paths relative to project root
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "evaluation")

LABEL_COLS = [
    "dry_eye_severity",
    "night_vision_disturbance",
    "ectasia_risk",
]

LABEL_DISPLAY_NAMES = {
    "dry_eye_severity"         : "Dry Eye Severity",
    "night_vision_disturbance" : "Night Vision Disturbance",
    "ectasia_risk"             : "Ectasia Risk",
}

# XGBoost hyperparameters
# These are reasonable defaults for a small clinical dataset.
# In a production setting these would be tuned via cross-validation.
XGBOOST_PARAMS = {
    "n_estimators"     : 200,
    "max_depth"        : 4,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "random_state"     : 42,
    "eval_metric"      : "logloss",
    "verbosity"        : 0,
}

RANDOM_STATE = 42


# ----------------------------------------------------------------
# Section 1: Load Preprocessed Data
# ----------------------------------------------------------------

def load_data():
    """Load the train and test splits produced by preprocess.py."""
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))

    print(f"Training set : {X_train.shape[0]} patients")
    print(f"Test set     : {X_test.shape[0]} patients")
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------
# Section 2: Train One Model Per Complication
# ----------------------------------------------------------------

def train_model(X_train, y_train, label):
    """
    Train an XGBoost classifier for a single complication label.

    Class imbalance handling:
        Ectasia risk is rare (~5% positive cases). To prevent the
        model from ignoring the minority class, sample weights are
        computed so that the minority class contributes proportionally
        more to the loss during training.

    Args:
        X_train : Feature matrix (training set)
        y_train : Full label DataFrame (training set)
        label   : Name of the target column to train on

    Returns:
        Fitted XGBClassifier
    """
    y = y_train[label].values
    n_classes = len(np.unique(y))

    # Configure objective based on number of classes
    if n_classes > 2:
        params = {**XGBOOST_PARAMS, "objective": "multi:softprob",
                  "num_class": n_classes}
    else:
        params = {**XGBOOST_PARAMS, "objective": "binary:logistic"}

    model = XGBClassifier(**params)

    # Compute sample weights to handle class imbalance
    sample_weights = compute_sample_weight(class_weight="balanced", y=y)

    model.fit(X_train, y, sample_weight=sample_weights)

    return model


# ----------------------------------------------------------------
# Section 3: Evaluate Model
# ----------------------------------------------------------------

def evaluate_model(model, X_test, y_test, label, report_lines):
    """
    Evaluate a trained model on the test set.
    Prints and saves:
        - Classification report (precision, recall, F1)
        - Confusion matrix plot
    """
    y_true = y_test[label].values
    y_pred = model.predict(X_test)

    display_name = LABEL_DISPLAY_NAMES[label]
    report = classification_report(y_true, y_pred, zero_division=0)

    print(f"\n  {display_name}")
    print(f"  {'-' * 40}")
    print(report)

    report_lines.append(f"\n{'=' * 50}")
    report_lines.append(f"{display_name}")
    report_lines.append(f"{'=' * 50}")
    report_lines.append(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {display_name}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()

    plot_path = os.path.join(
        REPORT_DIR, f"confusion_matrix_{label}.png"
    )
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to: {plot_path}")


# ----------------------------------------------------------------
# Section 4: Save Models
# ----------------------------------------------------------------

def save_model(model, label):
    """Save a fitted model to disk using pickle."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"model_{label}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved: {path}")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_data()

    report_lines = ["LASIK Risk Prediction — Model Evaluation Report\n"]
    models = {}

    print("\nTraining models...")
    for label in LABEL_COLS:
        display_name = LABEL_DISPLAY_NAMES[label]
        print(f"\n  Training: {display_name}")
        model = train_model(X_train, y_train, label)
        models[label] = model
        save_model(model, label)

    print("\nEvaluating models on test set...")
    for label in LABEL_COLS:
        evaluate_model(
            models[label], X_test, y_test, label, report_lines
        )

    # Save text report
    report_path = os.path.join(REPORT_DIR, "classification_reports.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nFull report saved to: {report_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()