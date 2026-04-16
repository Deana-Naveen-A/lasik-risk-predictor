"""
LASIK Risk Prediction Project
================================================================
File    : src/eda.py
Purpose : Exploratory Data Analysis (EDA) on the synthetic
          patient dataset generated in Step 1.

What this script produces:
    1. Basic dataset statistics
    2. Feature distributions (histograms)
    3. Complication class distributions (bar charts)
    4. Correlation heatmap between features
    5. Feature vs complication box plots for key relationships

Output:
    All plots are saved to: outputs/eda/

Usage:
    py src/eda.py
================================================================
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

DATA_PATH   = os.path.join("data", "lasik_dataset.csv")
OUTPUT_DIR  = os.path.join("outputs", "eda")

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

LABEL_DISPLAY_NAMES = {
    "dry_eye_severity"         : "Dry Eye Severity",
    "night_vision_disturbance" : "Night Vision Disturbance",
    "ectasia_risk"             : "Ectasia Risk",
}

# Consistent color palette across all plots
PALETTE = ["#2c7bb6", "#abd9e9", "#d7191c"]

# ----------------------------------------------------------------
# Utility
# ----------------------------------------------------------------

def setup():
    """Load data and create output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    return df


def save(fig, filename):
    """Save a figure to the output directory and close it."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ----------------------------------------------------------------
# Section 1: Dataset Summary
# ----------------------------------------------------------------

def print_summary(df):
    print("\n--- Dataset Summary ---")
    print(f"  Rows    : {df.shape[0]}")
    print(f"  Columns : {df.shape[1]}")

    print("\n--- Descriptive Statistics (Features) ---")
    print(df[FEATURE_COLS].describe().round(2).to_string())

    print("\n--- Label Distribution ---")
    for label in LABEL_COLS:
        print(f"\n  {LABEL_DISPLAY_NAMES[label]}:")
        counts = df[label].value_counts().sort_index()
        for val, count in counts.items():
            pct = count / len(df) * 100
            print(f"    Class {val}: {count} patients ({pct:.1f}%)")


# ----------------------------------------------------------------
# Section 2: Feature Distributions
# ----------------------------------------------------------------

def plot_feature_distributions(df):
    """
    Plot histogram for each continuous feature.
    Binary features (sex, autoimmune, medications) are shown
    as bar charts instead.
    """
    continuous = [
        "age", "corneal_thickness_um", "refractive_error_D",
        "tbut_seconds", "schirmer_mm", "pupil_diameter_mm",
        "osdi_score", "topography_index",
    ]
    binary = ["sex", "autoimmune_condition", "on_drying_medications"]

    # Continuous features
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Distribution of Continuous Pre-operative Features",
                 fontsize=14, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, col in enumerate(continuous):
        axes[i].hist(df[col], bins=30, color=PALETTE[0],
                     edgecolor="white", linewidth=0.5)
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
        axes[i].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    save(fig, "01_continuous_feature_distributions.png")

    # Binary features
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Distribution of Binary Pre-operative Features",
                 fontsize=14, fontweight="bold")
    binary_labels = {
        "sex"                  : ["Male", "Female"],
        "autoimmune_condition" : ["Absent", "Present"],
        "on_drying_medications": ["No", "Yes"],
    }

    for i, col in enumerate(binary):
        counts = df[col].value_counts().sort_index()
        axes[i].bar(binary_labels[col], counts.values,
                    color=[PALETTE[0], PALETTE[2]], edgecolor="white")
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_ylabel("Count")
        for j, v in enumerate(counts.values):
            axes[i].text(j, v + 5, str(v), ha="center", fontsize=9)

    fig.tight_layout()
    save(fig, "02_binary_feature_distributions.png")


# ----------------------------------------------------------------
# Section 3: Complication Class Distributions
# ----------------------------------------------------------------

def plot_label_distributions(df):
    """Bar chart for each complication label."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Post-operative Complication Class Distributions",
                 fontsize=14, fontweight="bold")

    label_class_names = {
        "dry_eye_severity"         : ["None", "Mild", "Chronic"],
        "night_vision_disturbance" : ["Absent", "Present"],
        "ectasia_risk"             : ["Low Risk", "High Risk"],
    }

    for i, label in enumerate(LABEL_COLS):
        counts = df[label].value_counts().sort_index()
        class_names = label_class_names[label]
        colors = PALETTE[:len(counts)]
        bars = axes[i].bar(class_names, counts.values,
                           color=colors, edgecolor="white")
        axes[i].set_title(LABEL_DISPLAY_NAMES[label], fontsize=11)
        axes[i].set_ylabel("Number of Patients")
        for bar, v in zip(bars, counts.values):
            pct = v / len(df) * 100
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 5,
                         f"{v}\n({pct:.1f}%)",
                         ha="center", fontsize=9)

    fig.tight_layout()
    save(fig, "03_label_distributions.png")


# ----------------------------------------------------------------
# Section 4: Correlation Heatmap
# ----------------------------------------------------------------

def plot_correlation_heatmap(df):
    """
    Heatmap of Pearson correlations between all features and labels.
    Helps identify which features are most predictive of each
    complication.
    """
    corr_cols = FEATURE_COLS + LABEL_COLS
    corr = df[corr_cols].corr().round(2)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature and Label Correlation Matrix",
                 fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    save(fig, "04_correlation_heatmap.png")


# ----------------------------------------------------------------
# Section 5: Key Feature vs Complication Box Plots
# ----------------------------------------------------------------

def plot_key_relationships(df):
    """
    Box plots showing the distribution of the most clinically
    important features across each complication class.
    These plots visually confirm that the label generation
    logic is clinically consistent.
    """
    relationships = [
        # (feature, label, title)
        ("tbut_seconds",         "dry_eye_severity",
         "TBUT vs Dry Eye Severity"),
        ("schirmer_mm",          "dry_eye_severity",
         "Schirmer Score vs Dry Eye Severity"),
        ("pupil_diameter_mm",    "night_vision_disturbance",
         "Pupil Diameter vs Night Vision Disturbance"),
        ("corneal_thickness_um", "ectasia_risk",
         "Corneal Thickness vs Ectasia Risk"),
        ("topography_index",     "ectasia_risk",
         "Topography Index vs Ectasia Risk"),
        ("osdi_score",           "dry_eye_severity",
         "OSDI Score vs Dry Eye Severity"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Key Clinical Feature vs Complication Relationships",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, (feature, label, title) in enumerate(relationships):
        groups = [
            df[df[label] == cls][feature].values
            for cls in sorted(df[label].unique())
        ]
        bp = axes[i].boxplot(
            groups,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[i].set_title(title, fontsize=10, fontweight="bold")
        axes[i].set_xlabel(f"{LABEL_DISPLAY_NAMES[label]} Class")
        axes[i].set_ylabel(feature.replace("_", " ").title())
        axes[i].set_xticklabels(sorted(df[label].unique()))

    fig.tight_layout()
    save(fig, "05_key_feature_vs_complication.png")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    print("Running Exploratory Data Analysis...")
    df = setup()

    print_summary(df)

    print("\nGenerating plots...")
    plot_feature_distributions(df)
    plot_label_distributions(df)
    plot_correlation_heatmap(df)
    plot_key_relationships(df)

    print(f"\nEDA complete. All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()