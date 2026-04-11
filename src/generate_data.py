"""
LASIK Risk Prediction Project

File    : src/generate_data.py
Purpose : Generate a synthetic dataset of pre-operative patient
          measurements and post-operative complication labels.

Complications predicted:
    1. Dry Eye Severity        - 0: None | 1: Mild | 2: Chronic
    2. Night Vision Disturbance - 0: Absent | 1: Present
    3. Ectasia Risk             - 0: Low | 1: High

Clinical basis:
    Feature distributions and label generation rules are derived
    from published LASIK complication literature, including FDA
    PROWL studies and peer-reviewed ophthalmology journals.

Note on synthetic data:
    Real patient data requires IRB approval and hospital access.
    Synthetic data generated from clinically validated statistical
    distributions is standard practice for proof-of-concept ML
    research and is acceptable for portfolio and early-stage
    academic work, provided it is clearly disclosed.

Output:
    data/lasik_dataset.csv

Usage:
    python src/generate_data.py

"""

import os
import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

N = 1000  # Number of synthetic patients


# Section 1: Feature Generation
# Each feature is sampled from a distribution whose parameters
# are grounded in published clinical ranges.


def generate_features(n):
    """
    Generate pre-operative clinical measurements for n patients.

    Returns a DataFrame with one row per patient and one column
    per clinical feature.
    """

    features = {

        # Demographic
        "age": np.random.normal(loc=32, scale=8, size=n).clip(20, 55),
        # Sex: 1 = Female, 0 = Male
        # Women have significantly higher rates of post-LASIK dry eye
        "sex": np.random.binomial(n=1, p=0.52, size=n),

        # Corneal measurements
        # Central corneal thickness in micrometers.
        # Normal range: 500-560 um. Thinner corneas carry higher
        # risk of ectasia after ablation.
        "corneal_thickness_um": np.random.normal(
            loc=535, scale=35, size=n
        ).clip(440, 620),

        # Refractive error in diopters (negative = myopic).
        # Higher absolute value means more corneal tissue must be
        # removed, reducing residual stromal bed thickness.
        "refractive_error_D": np.random.normal(
            loc=-3.5, scale=2.0, size=n
        ).clip(-10, -0.5),

        # Corneal topography irregularity index (KISA-composite proxy).
        # Higher values indicate more irregular corneal curvature,
        # a known precursor to post-LASIK ectasia.
        "topography_index": np.random.normal(
            loc=25, scale=15, size=n
        ).clip(0, 100),

        # Tear film measurements
        # Tear Break-Up Time (TBUT) in seconds.
        # Measures how quickly the tear film becomes unstable.
        # Normal: > 10 sec. Below 7 sec is considered at-risk.
        "tbut_seconds": np.random.normal(
            loc=9, scale=3, size=n
        ).clip(2, 20),

        # Schirmer Test score in mm.
        # Measures aqueous tear production over 5 minutes.
        # Normal: > 10 mm. Low values indicate reduced tear secretion.
        "schirmer_mm": np.random.normal(
            loc=12, scale=4, size=n
        ).clip(2, 25),

        # OSDI Score: Ocular Surface Disease Index (0-100).
        # A validated patient-reported dry eye questionnaire.
        # Higher scores indicate worse pre-existing dry eye symptoms.
        "osdi_score": np.random.normal(
            loc=18, scale=12, size=n
        ).clip(0, 100),

        # Scotopic (low-light) pupil diameter in mm.
        # Larger pupils increase the overlap between the pupil margin
        # and the ablation zone edge, causing halos and glare.
        "pupil_diameter_mm": np.random.normal(
            loc=5.5, scale=1.0, size=n
        ).clip(3.0, 8.5),

        # Systemic and medication flags
        # Pre-existing autoimmune condition (e.g. Sjogren syndrome,
        # lupus, rheumatoid arthritis). Impairs corneal healing and
        # significantly worsens post-operative dry eye.
        "autoimmune_condition": np.random.binomial(n=1, p=0.07, size=n),

        # Use of medications known to reduce tear secretion
        # (antihistamines, SSRIs, isotretinoin, anticholinergics).
        "on_drying_medications": np.random.binomial(n=1, p=0.15, size=n),
    }

    return pd.DataFrame(features)



# Section 2: Label Generation
# Labels are derived from features using clinically grounded
# linear scoring functions followed by thresholding.
# Random noise is added to simulate real-world variability.


def generate_labels(df):
    """
    Generate post-operative complication labels for each patient
    based on their pre-operative feature profile.

    Args:
        df : DataFrame of pre-operative features

    Returns:
        df : Same DataFrame with three label columns appended
    """

    n = len(df)

    # --- Label 1: Dry Eye Severity ---
    # Primary drivers: low TBUT, low Schirmer score, high OSDI,
    # female sex, autoimmune conditions, drying medications.
    dry_eye_score = (
        - 0.30 * df["tbut_seconds"]
        - 0.20 * df["schirmer_mm"]
        + 0.15 * df["osdi_score"]
        + 0.80 * df["sex"]
        + 1.50 * df["autoimmune_condition"]
        + 0.80 * df["on_drying_medications"]
        + np.random.normal(0, 1, n)
    )

    # Bin continuous score into three ordered severity classes
    df["dry_eye_severity"] = pd.cut(
        dry_eye_score,
        bins=[-np.inf, -1.5, 1.0, np.inf],
        labels=[0, 1, 2]
    ).astype(int)


    # --- Label 2: Night Vision Disturbance ---
    # Primary drivers: large scotopic pupil, high refractive error,
    # younger age.
    night_vision_score = (
        + 0.60 * df["pupil_diameter_mm"]
        - 0.30 * df["refractive_error_D"]
        - 0.02 * df["age"]
        + np.random.normal(0, 1, n)
    )

    night_vision_prob = 1 / (1 + np.exp(-(night_vision_score - 2)))
    df["night_vision_disturbance"] = (night_vision_prob > 0.5).astype(int)


    # --- Label 3: Ectasia Risk ---
    # Primary drivers: thin cornea, high refractive error (deep
    # ablation), irregular topography index.
    # Ectasia is rare (~5% in high-risk populations), so the
    # sigmoid intercept is set to reflect low base-rate prevalence.
    ectasia_score = (
        - 0.015 * df["corneal_thickness_um"]
        - 0.200 * df["refractive_error_D"]
        + 0.040 * df["topography_index"]
        + np.random.normal(0, 1, n)
    )

    # Use percentile-based threshold to reflect the clinically
    # observed ectasia prevalence of approximately 5% in screened
    # populations. Sigmoid thresholding is not used here because
    # the score range does not align with sigmoid sensitivity.
    ectasia_threshold = np.percentile(ectasia_score, 95)
    df["ectasia_risk"] = (ectasia_score >= ectasia_threshold).astype(int)

    return df



# Section 3: Save Dataset


def main():
    # Create output directory if it does not exist
    os.makedirs("data", exist_ok=True)

    print("Generating synthetic patient dataset...")

    df = generate_features(N)
    df = generate_labels(df)
    df = df.round(2)

    output_path = os.path.join("data", "lasik_dataset.csv")
    df.to_csv(output_path, index=False)

    # Summary report
    print(f"\nDataset saved to: {output_path}")
    print(f"Total patients  : {len(df)}")
    print(f"Total features  : {len(df.columns) - 3}")

    print("\nComplication distribution:")
    print(f"  Dry Eye Severity")
    print(f"    None    (0) : {(df['dry_eye_severity'] == 0).sum()} patients")
    print(f"    Mild    (1) : {(df['dry_eye_severity'] == 1).sum()} patients")
    print(f"    Chronic (2) : {(df['dry_eye_severity'] == 2).sum()} patients")
    print(f"  Night Vision Disturbance")
    print(f"    Present     : {df['night_vision_disturbance'].sum()} patients "
          f"({df['night_vision_disturbance'].mean() * 100:.1f}%)")
    print(f"  Ectasia Risk")
    print(f"    High Risk   : {df['ectasia_risk'].sum()} patients "
          f"({df['ectasia_risk'].mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()