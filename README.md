# NeuroScan AI — Parkinson’s Disease Detection System

## Overview

NeuroScan AI is a multi-modal Parkinson’s Disease detection and staging system developed using Machine Learning, Streamlit, Python, and biomedical signal analysis.

The system combines:

* Clinical symptom analysis
* Voice biomarker analysis
* Motor tapping inference
* Multi-modal fusion AI

The goal of the project is to assist in early Parkinson’s Disease screening and classify disease severity into:

1. Healthy
2. Mild
3. Moderate
4. Severe

This project is intended for research and educational purposes only and is not a certified medical diagnostic device.

---

# Features

## Clinical Analysis

Uses:

* UPDRS score
* Tremor severity
* Rigidity
* Speech difficulty
* Facial expression
* Gait instability
* Postural stability
* Disease duration
* Age

The clinical model is the strongest modality in the system.

---

## Live Voice Recording

The patient records a sustained:

“Aaaaaah” vowel sound for 5–7 seconds.

The system extracts Parkinson’s-related voice biomarkers such as:

* HNR
* NHR
* PPE
* RPDE
* Pitch variation
* Jitter
* Shimmer

These biomarkers help detect subtle neurological changes.

---

## Motor Tapping Inference

The system uses stage-profile motor inference to simulate tapping behavior based on predicted severity.

This improves robustness of the final fusion prediction.

---

## Multi-Modal AI Fusion

The final prediction combines:

| Model          | Weight |
| -------------- | ------ |
| Clinical Model | 55–70% |
| Tapping Model  | 20–30% |
| Voice Model    | 25%    |

If voice data is unavailable, weights are redistributed automatically.

---

# Technologies Used

## Frontend

* Streamlit
* HTML
* CSS
* JavaScript

## Backend

* Python

## Machine Learning

* XGBoost
* Random Forest
* Scikit-learn

## Audio Processing

* Librosa
* NumPy
* SciPy

## Data Handling

* Pandas
* Joblib

---

# Folder Structure

```text
Major Project/
│
├── app.py
├── fusion_model.py
├── voice_extractor.py
├── train_voice_model.py
├── train_tapping.py
├── evaluate_neuroscan.py
├── test_fusion.py
│
├── parkinson_model.pkl
├── scaler.pkl
├── features.pkl
│
├── tap_model.pkl
├── tap_scaler.pkl
├── tap_features.pkl
├── tap_stage_profiles.pkl
│
├── voice_model.pkl
├── voice_scaler.pkl
├── voice_features.pkl
│
├── requirements.txt
└── README.md
```

---

# Machine Learning Models

## 1. Clinical Model

### Algorithm

XGBoost Classifier

### Input Features

* Age
* Disease Duration
* UPDRS Score
* Speech Difficulty
* Facial Expression
* Tremor
* Rigidity
* Gait
* Postural Stability

### Purpose

Predict Parkinson’s severity using clinical symptoms.

### Output

* Mild
* Moderate
* Severe

Healthy cases are handled separately using rule-based screening.

---

## 2. Voice Model

### Algorithm

Random Forest Classifier

### Input Features

22 MDVP acoustic biomarkers.

### Purpose

Detect Parkinson’s-related vocal abnormalities.

### Important Biomarkers

* HNR
* NHR
* PPE
* RPDE
* Jitter
* Shimmer

### Why Voice Matters

Voice changes often appear early in Parkinson’s Disease.

This helps detect:

* Healthy vs Mild
* Mild vs Moderate

cases more effectively.

---

## 3. Tapping Model

### Algorithm

Random Forest Classifier

### Purpose

Estimate motor impairment patterns.

This implementation uses stage-profile inference instead of live tapping sensors.

---

# Fusion Architecture

```text
Clinical Features
        ↓
Clinical Model
        ↓

Voice Recording
        ↓
Voice Biomarker Extraction
        ↓
Voice Model
        ↓

Tapping Stage Profiles
        ↓
Tapping Model
        ↓

Weighted Fusion Engine
        ↓
Final Stage Prediction
```

---

# Parkinson’s Stages

## Healthy

No major Parkinson’s indicators detected.

---

## Stage 1 — Mild

* Early symptoms
* Slight tremor
* Mild speech issues
* Minimal functional impact

---

## Stage 2 — Moderate

* Bilateral symptoms
* Walking difficulty
* Increased rigidity
* Daily activities affected

---

## Stage 3 — Severe

* Advanced motor impairment
* Postural instability
* High fall risk
* Significant mobility issues

---

# Installation Guide



## Step 1 — Open Project Folder

```bash
cd YOUR_REPOSITORY
```

---

## Step 2 — Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is missing:

```bash
pip install streamlit pandas numpy scikit-learn xgboost librosa scipy soundfile joblib imbalanced-learn
```

---

# How to Run the Project

Run Streamlit app:

```bash
streamlit run app.py
```

The app will open automatically in your browser.

Usually at:

```text
http://localhost:8501
```

---

# How to Use the System

## Step 1 — Enter Patient Information

Provide:

* Age
* Disease Duration
* UPDRS Score

---

## Step 2 — Select Symptom Severity

Rate:

* Speech
* Tremor
* Rigidity
* Gait
* Postural stability

using the slider controls.

---

## Step 3 — Record Voice

Click:

```text
Start Recording
```

Say:

```text
“Aaaaaah”
```

for 5–7 seconds.

Then click:

```text
Stop Recording
```

Alternatively, upload a `.wav` audio file.

---

## Step 4 — Run Analysis

Click:

```text
Run NeuroScan Analysis
```

The system will:

1. Extract voice biomarkers
2. Run all AI models
3. Fuse predictions
4. Display Parkinson’s stage
5. Show confidence scores
6. Show model contributions
7. Display clinical insights

---

# Expected Outputs

The application shows:

* Final Parkinson’s stage
* Confidence score
* Stage probabilities
* Clinical metrics
* Model contributions
* Voice biomarkers
* AI fusion details

---

# Important Files

## app.py

Main Streamlit frontend application.

Handles:

* UI
* Voice recording
* Prediction display
* Result visualization

---

## fusion_model.py

Core fusion engine.

Combines:

* Clinical prediction
* Voice prediction
* Tapping prediction

into a final stage.

---

## voice_extractor.py

Extracts 22 voice biomarkers from recorded audio.

---

## train_voice_model.py

Trains the voice Random Forest classifier.

---

## train_tapping.py

Trains the tapping model.

---

## evaluate_neuroscan.py

Evaluates overall model performance.

---

# Accuracy

Approximate project accuracies:

| Model          | Accuracy |
| -------------- | -------- |
| Clinical Model | ~99%     |
| Voice Model    | ~85%     |
| Tapping Model  | ~76%     |
| Fusion System  | ~91%     |

Fusion improves robustness and reliability compared to single-modality systems.

---



# Research Significance

This project demonstrates how AI and biomedical signal processing can help:

* detect Parkinson’s Disease earlier
* improve diagnostic support
* combine multiple modalities
* create accessible healthcare screening systems

---

# Disclaimer

NeuroScan AI is developed for:

* academic research
* educational learning
* project demonstration

It should NOT replace professional neurological diagnosis.

Always consult a certified medical professional.

---

# Author

Developed as a Major Project using:

* Python
* Streamlit
* Machine Learning
* Biomedical Signal Processing
* Multi-Modal AI Fusion
