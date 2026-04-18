"""
LASIK Risk Prediction Project
================================================================
File    : src/preprocess.py
Purpose : Load the raw dataset, apply preprocessing steps, and
          save train/test splits ready for model training.

Steps performed:
    1. Load raw dataset
    2. Separate features and labels
    3. Scale continuous features using StandardScaler
    4. Split into train (80%) and test (20%) sets
    5. Save all splits to disk for use in training

Output:
    data/X_train.csv
    data/X_test.csv
    data/y_train.csv
    data/y_test.csv
    data/scaler.pkl        (fitted scaler, saved for inference)

Usage:
    py src/preprocess.py
================================================================
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Always resolve paths relative to this file's location
# so the script works regardless of where it is called from
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

DATA_PATH   = os.path.join(PROJECT_ROOT, "data", "lasik_dataset.csv")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "data")

FEATURE_COLS = [
    "age",
    "sex",
    "corneal_thickness_um",
    "refractive_error_D",
    "tbut_seconds",
    "schirmer_mm",
    "pupil_diameter_mm",
    "osdi_score",
    "topography_index",
    "autoimmune_condition",
    "on_drying_medications",
]

LABEL_COLS = [
    "dry_eye_severity",
    "night_vision_disturbance",
    "ectasia_risk",
]

# These features are continuous and will be scaled.
# Binary features (sex, autoimmune_condition, on_drying_medications)
# are left as-is since they are already on a 0/1 scale.
CONTINUOUS_COLS = [
    "age",
    "corneal_thickness_um",
    "refractive_error_D",
    "tbut_seconds",
    "schirmer_mm",
    "pupil_diameter_mm",
    "osdi_score",
    "topography_index",
]

TEST_SIZE    = 0.20   # 20% of data held out for testing
RANDOM_STATE = 42     # Ensures reproducibility


# ----------------------------------------------------------------
# Section 1: Load Data
# ----------------------------------------------------------------

def load_data(path):
    """Load the CSV dataset and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ----------------------------------------------------------------
# Section 2: Check for Missing Values
# ----------------------------------------------------------------

def check_missing(df):
    """
    Report any missing values in the dataset.
    Since this is synthetic data there should be none, but this
    step is included as good practice for when real data is used.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values detected.")
    else:
        print("Missing values found:")
        print(missing.to_string())

    return df


# ----------------------------------------------------------------
# Section 3: Split Features and Labels
# ----------------------------------------------------------------

def split_features_labels(df):
    """Separate input features from output labels."""
    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COLS].copy()
    print(f"Features shape : {X.shape}")
    print(f"Labels shape   : {y.shape}")
    return X, y


# ----------------------------------------------------------------
# Section 4: Scale Continuous Features
# ----------------------------------------------------------------

def scale_features(X_train, X_test):
    """
    Fit a StandardScaler on training data and apply it to both
    train and test sets.

    Important: The scaler is fitted ONLY on training data to
    prevent data leakage. The same fitted scaler is then applied
    to the test set.

    StandardScaler transforms each feature to have:
        mean  = 0
        std   = 1

    This ensures that features with large ranges (e.g. corneal
    thickness in micrometers) do not dominate features with small
    ranges (e.g. TBUT in seconds) during model training.

    Binary features are excluded from scaling since they are
    already on a 0/1 scale.
    """
    scaler = StandardScaler()

    X_train[CONTINUOUS_COLS] = scaler.fit_transform(
        X_train[CONTINUOUS_COLS]
    )
    X_test[CONTINUOUS_COLS] = scaler.transform(
        X_test[CONTINUOUS_COLS]
    )

    print("Continuous features scaled using StandardScaler.")
    return X_train, X_test, scaler


# ----------------------------------------------------------------
# Section 5: Save Outputs
# ----------------------------------------------------------------

def save_splits(X_train, X_test, y_train, y_test, scaler):
    """Save train/test splits and the fitted scaler to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"),  index=False)

    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved:")
    print(f"  data/X_train.csv  — {X_train.shape[0]} rows")
    print(f"  data/X_test.csv   — {X_test.shape[0]} rows")
    print(f"  data/y_train.csv  — {y_train.shape[0]} rows")
    print(f"  data/y_test.csv   — {y_test.shape[0]} rows")
    print(f"  data/scaler.pkl   — fitted StandardScaler")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    print("Running preprocessing pipeline...\n")

    df = load_data(DATA_PATH)
    df = check_missing(df)

    X, y = split_features_labels(df)

    # Split before scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"\nTrain set : {X_train.shape[0]} patients")
    print(f"Test set  : {X_test.shape[0]} patients")

    X_train, X_test, scaler = scale_features(X_train, X_test)

    save_splits(X_train, X_test, y_train, y_test, scaler)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()