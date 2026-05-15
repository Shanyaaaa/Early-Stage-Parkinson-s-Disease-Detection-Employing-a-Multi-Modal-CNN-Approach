"""
fusion_model.py — FINAL VERSION
Predicts: 0=Healthy | 1=Mild | 2=Moderate | 3=Severe

Tap model uses stage-profile inference (no live sensor needed).
Voice model is optional (upload CSV of MDVP features).
"""

import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
#  LOAD MODELS
# ------------------------------------------------------------------ #
clinical_model     = joblib.load("parkinson_model.pkl")
clinical_scaler    = joblib.load("scaler.pkl")
clinical_features  = joblib.load("features.pkl")

tap_model          = joblib.load("tap_model.pkl")
tap_scaler         = joblib.load("tap_scaler.pkl")
tap_features       = joblib.load("tap_features.pkl")
tap_stage_profiles = joblib.load("tap_stage_profiles.pkl")

voice_model        = joblib.load("voice_model.pkl")
voice_scaler       = joblib.load("voice_scaler.pkl")
voice_features     = joblib.load("voice_features.pkl")


def _normalize(p):
    s = np.sum(p)
    return p / s if s > 1e-10 else np.ones_like(p) / len(p)


def _pad_proba(proba, model, n=3):
    if len(proba) == n:
        return np.array(proba, dtype=float)
    full = np.zeros(n)
    for i, c in enumerate(model.classes_):
        if c < n:
            full[c] = proba[i]
    return full


def _tap_from_clinical(p_clinical):
    vec = sum(float(p_clinical[i]) * tap_stage_profiles[i] for i in range(3))
    df  = pd.DataFrame([vec], columns=tap_features)
    p   = tap_model.predict_proba(tap_scaler.transform(df))[0]
    return _pad_proba(p, tap_model)

def _is_healthy(clinical_input):
    _, duration, updrs, speech, facial, tremor, rigidity, gait, postural = clinical_input
    motor_sum = speech + facial + tremor + rigidity + gait + postural
    return updrs < 10 and motor_sum <= 2 and duration < 1


# --- Change this section inside fusion_model.py ---

def predict_stage(clinical_input, voice_input=None):
    """
    clinical_input : list[9]
    voice_input    : list[22] or None
    """
    age, duration, updrs = clinical_input[0], clinical_input[1], clinical_input[2]

    # Validation
    if not (1 <= age <= 120):
        raise ValueError(f"Age must be 1–120 (got {age})")
    if updrs < 0 or updrs > 132:
        raise ValueError(f"UPDRS must be 0–132 (got {updrs})")

    # 1. Clinical Base Probability Calculations
    cdf      = pd.DataFrame([clinical_input], columns=clinical_features)
    p_clin   = _normalize(_pad_proba(clinical_model.predict_proba(clinical_scaler.transform(cdf))[0], clinical_model))

    # 2. Tap Profile Inference calculations
    p_tap    = _normalize(_tap_from_clinical(p_clin))

    # 3. Voice Data Evaluation Gate
    use_voice = (
        voice_input is not None
        and len(voice_input) == len(voice_features)
        and not all(float(v) == 0.0 for v in voice_input)
    )
    
    if use_voice:
        vdf    = pd.DataFrame([voice_input], columns=voice_features)
        p_voc  = _normalize(_pad_proba(voice_model.predict_proba(voice_scaler.transform(vdf))[0], voice_model))
    else:
        p_voc  = None

    # 4. OVERRIDE RULE: Apply Healthy Screening weights without skipping voice data
    if _is_healthy(clinical_input):
        # Force clinical probabilities to reflect a clean profile safely
        p_clin = np.array([1.0, 0.0, 0.0])
        p_tap  = np.array([1.0, 0.0, 0.0])
        
        if use_voice:
            # Keep the voice model active, but prioritize the clear clinical profile
            w_c, w_t, w_v = 0.60, 0.20, 0.20
            tap_mode = "healthy-screened + voice active"
        else:
            w_c, w_t, w_v = 0.70, 0.30, 0.00
            tap_mode = "healthy-screened"
    else:
        # Standard processing weights for symptomatic profiles
        if use_voice:
            w_c, w_t, w_v = 0.55, 0.20, 0.25
            tap_mode = "stage-profile inference"
        else:
            w_c, w_t, w_v = 0.70, 0.30, 0.00
            tap_mode = "stage-profile inference"

    # 5. Final Modal Fusion Calculations
    final = _normalize(w_c * p_clin + w_t * p_tap + (w_v * p_voc if use_voice else 0))
    stage = int(np.argmax(final)) + 1
    max_p = float(np.max(final))

    debug = {
        "stage_label":  ["Mild","Moderate","Severe"][stage-1] if max_p > 0.0 else "Healthy",
        "clinical_prob": p_clin.tolist(),
        "tap_prob":      p_tap.tolist(),
        "voice_prob":    p_voc.tolist() if use_voice else None,
        "final_prob":    final.tolist(),
        "weights":       {"clinical": w_c, "tap": w_t, "voice": w_v},
        "voice_used":    use_voice,
        "tap_mode":      tap_mode,
        "confident":     max_p >= 0.50,
        "max_prob":      max_p,
    }
    
    # If it is structurally healthy, classify as stage 0
    if _is_healthy(clinical_input) and not (use_voice and np.argmax(p_voc) > 0):
        return 0, final.tolist(), debug
        
    return stage, final.tolist(), debug