"""
LASIK Risk Prediction Project
================================================================
File    : src/explain.py
Purpose : Generate SHAP (SHapley Additive exPlanations) plots
          for each trained model to explain which pre-operative
          features drive each complication risk prediction.

Why explainability matters in clinical ML:
    A model that only outputs a risk score is not useful to a
    clinician. SHAP explains the contribution of each feature
    to each individual prediction, making the model transparent
    and interpretable. This is a requirement for any ML system
    intended for medical decision support.

Plots generated:
    1. Summary plot   — global feature importance across all patients
    2. Bar plot       — mean absolute SHAP value per feature
    3. Waterfall plot — single patient explanation (patient index 0)

Output:
    outputs/shap/shap_summary_<label>.png
    outputs/shap/shap_bar_<label>.png
    outputs/shap/shap_waterfall_<label>.png

Usage:
    py src/explain.py
================================================================
"""

import os
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Resolve paths relative to project root
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "shap")

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

FEATURE_DISPLAY_NAMES = {
    "age"                   : "Age",
    "sex"                   : "Sex (Female=1)",
    "corneal_thickness_um"  : "Corneal Thickness (um)",
    "refractive_error_D"    : "Refractive Error (D)",
    "tbut_seconds"          : "Tear Break-Up Time (s)",
    "schirmer_mm"           : "Schirmer Score (mm)",
    "pupil_diameter_mm"     : "Pupil Diameter (mm)",
    "osdi_score"            : "OSDI Score",
    "topography_index"      : "Topography Index",
    "autoimmune_condition"  : "Autoimmune Condition",
    "on_drying_medications" : "Drying Medications",
}


# ----------------------------------------------------------------
# Section 1: Load Data and Models
# ----------------------------------------------------------------

def load_data():
    """Load the test set features."""
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    # Rename columns to display-friendly names for plots
    X_test = X_test.rename(columns=FEATURE_DISPLAY_NAMES)
    return X_test


def load_model(label):
    """Load a saved model from disk."""
    path = os.path.join(MODEL_DIR, f"model_{label}.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# ----------------------------------------------------------------
# Section 2: Compute SHAP Values
# ----------------------------------------------------------------

def compute_shap_values(model, X_test):
    """
    Compute SHAP values using TreeExplainer.

    TreeExplainer is used because XGBoost is a tree-based model.
    It is fast and exact for tree ensembles.

    For multiclass models, SHAP returns values for each class.
    We use the values for class 1 (mild) as a representative
    middle-ground explanation for dry eye severity.

    Returns:
        explainer    : fitted SHAP explainer
        shap_values  : SHAP value array (n_samples x n_features)
        explanation  : SHAP Explanation object for waterfall plots
    """
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test)

    shap_values = explanation.values

    # For multiclass output, select class 1 values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
        explanation = explanation[:, :, 1]

    return explainer, shap_values, explanation


# ----------------------------------------------------------------
# Section 3: Generate Plots
# ----------------------------------------------------------------

def plot_summary(shap_values, X_test, label):
    """
    Beeswarm summary plot.
    Each dot is one patient. Color shows feature value (red = high,
    blue = low). X axis shows SHAP value (impact on prediction).
    Features are ranked by mean absolute SHAP value.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        plot_type="dot",
        show=False,
        plot_size=None,
    )
    plt.title(
        f"SHAP Summary — {LABEL_DISPLAY_NAMES[label]}",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"shap_summary_{label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_bar(shap_values, X_test, label):
    """
    Bar plot of mean absolute SHAP values.
    Shows overall feature importance across all patients.
    This is the easiest plot to interpret at a glance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test,
        plot_type="bar",
        show=False,
        plot_size=None,
    )
    plt.title(
        f"Feature Importance (Mean |SHAP|) — {LABEL_DISPLAY_NAMES[label]}",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"shap_bar_{label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_waterfall(explanation, label, patient_index=0):
    """
    Waterfall plot for a single patient.
    Shows exactly which features pushed the prediction up or down
    for that specific patient. This is the per-patient explanation
    that makes the model clinically useful.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(explanation[patient_index], show=False)
    plt.title(
        f"Patient-Level Explanation — {LABEL_DISPLAY_NAMES[label]}"
        f" (Patient {patient_index})",
        fontsize=12, fontweight="bold", pad=15
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"shap_waterfall_{label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading test data...")
    X_test = load_data()

    for label in LABEL_COLS:
        display_name = LABEL_DISPLAY_NAMES[label]
        print(f"\nGenerating SHAP explanations: {display_name}")

        model = load_model(label)
        explainer, shap_values, explanation = compute_shap_values(
            model, X_test
        )

        plot_summary(shap_values, X_test, label)
        plot_bar(shap_values, X_test, label)
        plot_waterfall(explanation, label, patient_index=0)

    print(f"\nSHAP explainability complete.")
    print(f"All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
