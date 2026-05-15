# fusion_model.py

import joblib
import numpy as np
import pandas as pd


# -------- LOAD MODELS --------
clinical_model = joblib.load("parkinson_model.pkl")
clinical_scaler = joblib.load("scaler.pkl")
clinical_features = joblib.load("features.pkl")   # 🔥 IMPORTANT

tap_model = joblib.load("tap_model.pkl")
tap_scaler = joblib.load("tap_scaler.pkl")
tap_features = joblib.load("tap_features.pkl")

voice_model = joblib.load("voice_model.pkl")
voice_scaler = joblib.load("voice_scaler.pkl")
voice_features = joblib.load("voice_features.pkl")


def safe_normalize(p):
    """Avoid divide-by-zero"""
    total = np.sum(p)
    return p / total if total != 0 else np.ones_like(p) / len(p)


def predict_stage(clinical_input, voice_input, tap_input):

    # -------- VALIDATION --------
    if len(clinical_input) != len(clinical_features):
        raise ValueError("Clinical input length mismatch")

    if len(tap_input) != len(tap_features):
        raise ValueError("Tap input length mismatch")

    if voice_input is not None and len(voice_input) != len(voice_features):
        raise ValueError("Voice input length mismatch")

    if sum(clinical_input) == 0:
        raise ValueError("Clinical input cannot be all zeros")

    # -------- CREATE DATAFRAMES (FIX FEATURE ALIGNMENT) --------
    clinical_df = pd.DataFrame([clinical_input], columns=clinical_features)
    tap_df = pd.DataFrame([tap_input], columns=tap_features)

    # -------- SCALE --------
    clinical_scaled = clinical_scaler.transform(clinical_df)
    tap_scaled = tap_scaler.transform(tap_df)

    # -------- MODEL PROBABILITIES --------
    p_clinical = clinical_model.predict_proba(clinical_scaled)[0]
    p_tap = tap_model.predict_proba(tap_scaled)[0]

    # -------- VOICE HANDLING --------
    if voice_input is None or np.all(np.array(voice_input) == 0):
        p_voice_3 = np.array([1.0, 0.0, 0.0])
        voice_weight = 0.0
    else:
        voice_df = pd.DataFrame([voice_input], columns=voice_features)
        voice_scaled = voice_scaler.transform(voice_df)

        p_voice = voice_model.predict_proba(voice_scaled)[0]

        # Convert binary → 3 class
        p_voice_3 = np.array([
            1 - p_voice[1],
            p_voice[1] * 0.7,
            p_voice[1] * 0.3
        ])
        voice_weight = 0.15

    # -------- NORMALIZE --------
    p_clinical = safe_normalize(p_clinical)
    p_tap = safe_normalize(p_tap)
    p_voice_3 = safe_normalize(p_voice_3)

    # -------- DYNAMIC WEIGHTS --------
    clinical_weight = 0.55
    tap_weight = 0.30

    if voice_weight == 0:
        # redistribute voice weight
        clinical_weight += 0.10
        tap_weight += 0.05

    # -------- FUSION --------
    final_prob = (
        clinical_weight * p_clinical +
        tap_weight * p_tap +
        voice_weight * p_voice_3
    )

    final_stage = int(np.argmax(final_prob) + 1)

    # -------- DEBUG INFO --------
    debug = {
        "clinical_prob": p_clinical.tolist(),
        "tap_prob": p_tap.tolist(),
        "voice_prob": p_voice_3.tolist(),
        "weights": {
            "clinical": clinical_weight,
            "tap": tap_weight,
            "voice": voice_weight
        },
        "max_prob": float(np.max(final_prob)),
        "confident": bool(np.max(final_prob) > 0.5)
    }

    return final_stage, final_prob.tolist(), debug




