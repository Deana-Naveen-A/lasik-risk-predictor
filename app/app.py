"""
LASIK Risk Prediction Project
================================================================
File    : app/app.py
Purpose : Streamlit web application for predicting post-operative
          LASIK complication risks from pre-operative patient
          measurements.

The app allows a user to input clinical measurements via sliders
and dropdowns, then displays:
    1. Risk scores for each of the three complications
    2. SHAP waterfall explanation for each prediction showing
       which features drove the risk up or down for that patient

Usage:
    streamlit run app/app.py
================================================================
"""

import os
import pickle
import sys
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit as st

# Resolve project root so imports work regardless of working directory
APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

# ----------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------

st.set_page_config(
    page_title="LASIK Risk Predictor",
    page_icon="",
    layout="wide",
)

# ----------------------------------------------------------------
# Load Models and Scaler
# ----------------------------------------------------------------

@st.cache_resource
def load_models():
    """
    Load all three trained models and the fitted scaler from disk.
    Cached so they are only loaded once per session.
    """
    models = {}
    for label in ["dry_eye_severity", "night_vision_disturbance", "ectasia_risk"]:
        path = os.path.join(MODEL_DIR, f"model_{label}.pkl")
        with open(path, "rb") as f:
            models[label] = pickle.load(f)

    scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return models, scaler


# ----------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------

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


def preprocess_input(inputs, scaler):
    """
    Convert raw user inputs into a scaled feature vector
    matching the format used during model training.
    """
    df = pd.DataFrame([inputs], columns=FEATURE_COLS)
    df[CONTINUOUS_COLS] = scaler.transform(df[CONTINUOUS_COLS])
    return df


# ----------------------------------------------------------------
# SHAP Explanation
# ----------------------------------------------------------------

def get_shap_explanation(model, X_input, label):
    """
    Compute SHAP values for a single patient input and return
    a waterfall plot as a matplotlib figure.
    """
    explainer   = shap.TreeExplainer(model)
    explanation = explainer(X_input)

    # For multiclass models, use class 1 (mild/present)
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]

    # Rename features for display
    explanation.feature_names = [
        FEATURE_DISPLAY_NAMES.get(f, f)
        for f in FEATURE_COLS
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation[0], show=False)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------
# Risk Display Helpers
# ----------------------------------------------------------------

def dry_eye_label(pred):
    return {0: "None", 1: "Mild", 2: "Chronic"}.get(pred, "Unknown")

def binary_label(pred):
    return "Present" if pred == 1 else "Absent"

def risk_color(pred, label):
    if label == "dry_eye_severity":
        return ["green", "orange", "red"][pred]
    return "red" if pred == 1 else "green"


# ----------------------------------------------------------------
# Main App
# ----------------------------------------------------------------

def main():
    models, scaler = load_models()

    # Header
    st.title("LASIK Post-operative Risk Predictor")
    st.markdown(
        """
        This tool predicts the risk of three post-operative complications
        following LASIK refractive surgery based on pre-operative clinical
        measurements. Enter the patient's measurements in the panel on the
        left and click **Predict** to generate a risk assessment.

        > **Disclaimer:** This is a proof-of-concept research tool built on
        > synthetic data. It is not validated for clinical use and should not
        > inform real medical decisions.
        """
    )

    st.divider()

    # Sidebar — Patient Input
    st.sidebar.header("Patient Pre-operative Measurements")

    age = st.sidebar.slider(
        "Age (years)", min_value=20, max_value=55, value=32
    )
    sex = st.sidebar.selectbox(
        "Sex", options=["Male", "Female"]
    )
    corneal_thickness = st.sidebar.slider(
        "Central Corneal Thickness (um)", min_value=440, max_value=620, value=535
    )
    refractive_error = st.sidebar.slider(
        "Refractive Error (Diopters)", min_value=-10.0, max_value=-0.5,
        value=-3.5, step=0.25
    )
    tbut = st.sidebar.slider(
        "Tear Break-Up Time — TBUT (seconds)", min_value=2, max_value=20, value=9
    )
    schirmer = st.sidebar.slider(
        "Schirmer Test Score (mm)", min_value=2, max_value=25, value=12
    )
    pupil = st.sidebar.slider(
        "Scotopic Pupil Diameter (mm)", min_value=3.0, max_value=8.5,
        value=5.5, step=0.1
    )
    osdi = st.sidebar.slider(
        "OSDI Score (0-100)", min_value=0, max_value=100, value=18
    )
    topo = st.sidebar.slider(
        "Topography Index", min_value=0, max_value=100, value=25
    )
    autoimmune = st.sidebar.selectbox(
        "Pre-existing Autoimmune Condition", options=["No", "Yes"]
    )
    drying_meds = st.sidebar.selectbox(
        "On Drying Medications", options=["No", "Yes"]
    )

    predict_button = st.sidebar.button("Predict Risk", type="primary")

    # Prediction
    if predict_button:
        inputs = {
            "age"                   : age,
            "sex"                   : 1 if sex == "Female" else 0,
            "corneal_thickness_um"  : corneal_thickness,
            "refractive_error_D"    : refractive_error,
            "tbut_seconds"          : tbut,
            "schirmer_mm"           : schirmer,
            "pupil_diameter_mm"     : pupil,
            "osdi_score"            : osdi,
            "topography_index"      : topo,
            "autoimmune_condition"  : 1 if autoimmune == "Yes" else 0,
            "on_drying_medications" : 1 if drying_meds == "Yes" else 0,
        }

        X_input = preprocess_input(inputs, scaler)

        # Predictions
        dry_eye_pred    = models["dry_eye_severity"].predict(X_input)[0]
        night_vis_pred  = models["night_vision_disturbance"].predict(X_input)[0]
        ectasia_pred    = models["ectasia_risk"].predict(X_input)[0]

        # Risk Summary
        st.subheader("Risk Assessment")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Dry Eye Severity", dry_eye_label(dry_eye_pred))
            if dry_eye_pred == 0:
                st.success("Low risk")
            elif dry_eye_pred == 1:
                st.warning("Moderate risk")
            else:
                st.error("High risk")

        with col2:
            st.metric("Night Vision Disturbance", binary_label(night_vis_pred))
            if night_vis_pred == 0:
                st.success("Low risk")
            else:
                st.error("High risk")

        with col3:
            st.metric("Ectasia Risk", binary_label(ectasia_pred))
            if ectasia_pred == 0:
                st.success("Low risk")
            else:
                st.error("High risk")

        st.divider()

        # SHAP Explanations
        st.subheader("Feature-Level Explanations (SHAP)")
        st.markdown(
            "The plots below show which pre-operative measurements "
            "contributed most to each risk prediction for this patient. "
            "Red bars increase risk, blue bars decrease risk."
        )

        tab1, tab2, tab3 = st.tabs([
            "Dry Eye Severity",
            "Night Vision Disturbance",
            "Ectasia Risk",
        ])

        with tab1:
            fig = get_shap_explanation(
                models["dry_eye_severity"], X_input, "dry_eye_severity"
            )
            st.pyplot(fig)
            plt.close()

        with tab2:
            fig = get_shap_explanation(
                models["night_vision_disturbance"], X_input,
                "night_vision_disturbance"
            )
            st.pyplot(fig)
            plt.close()

        with tab3:
            fig = get_shap_explanation(
                models["ectasia_risk"], X_input, "ectasia_risk"
            )
            st.pyplot(fig)
            plt.close()

    else:
        st.info("Enter patient measurements in the left panel and click Predict Risk.")


if __name__ == "__main__":
    main()
