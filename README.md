## Input Features

| Feature | Description | Clinical Relevance |
|---|---|---|
| Age | Patient age in years | Younger patients have higher night vision disturbance rates |
| Sex | Male / Female | Women have significantly higher post-LASIK dry eye rates |
| Central Corneal Thickness | Micrometers (um) | Thinner corneas carry higher ectasia risk |
| Refractive Error | Diopters (D) | Higher prescriptions require deeper ablation |
| Tear Break-Up Time (TBUT) | Seconds | Below 7s indicates pre-existing dry eye risk |
| Schirmer Test Score | Millimeters (mm) | Measures aqueous tear production |
| Scotopic Pupil Diameter | Millimeters (mm) | Large pupils increase halo and glare risk |
| OSDI Score | 0 to 100 | Patient-reported dry eye symptom severity |
| Topography Index | Composite score | Irregular corneal curvature signals ectasia risk |
| Autoimmune Condition | Binary | Impairs healing and worsens dry eye |
| Drying Medications | Binary | Antihistamines, SSRIs, isotretinoin reduce tear secretion |

---

## Tech Stack

- Python 3.12
- XGBoost — gradient boosted tree classifiers
- scikit-learn — preprocessing, evaluation, sample weighting
- SHAP — model explainability
- Streamlit — interactive frontend
- Matplotlib / Seaborn — visualizations

---

## Setup and Usage

1. Clone the repository
git clone https://github.com/Deana-Naveen-A/lasik-risk-predictor.git
cd lasik-risk-predictor

2. Install dependencies
pip install -r requirements.txt

3. Generate synthetic dataset
py src/generate_data.py

4. Run preprocessing
py src/preprocess.py

5. Train models
py src/train.py

6. Generate SHAP plots
py src/explain.py

7. Launch the app
streamlit run app/app.py

---

## Data

This project uses synthetically generated data. Real patient data for LASIK outcomes requires IRB approval and institutional access. The synthetic dataset was generated from clinically validated statistical distributions derived from published LASIK complication literature, including the FDA PROWL-1 and PROWL-2 patient-reported outcomes studies.

All results and accuracy metrics reflect model performance on synthetic data and are not indicative of real-world clinical performance.

---

## Disclaimer

This is a proof-of-concept research project built for academic and portfolio purposes. It is not validated for clinical use and must not be used to inform real medical decisions.

---

## Author

Deana Naveen
B.E. Computer Science and Engineering (AI/ML)
PES University, Bangalore
