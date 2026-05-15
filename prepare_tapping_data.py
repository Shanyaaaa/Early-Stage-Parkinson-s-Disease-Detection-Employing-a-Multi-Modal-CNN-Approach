"""
preprocess_tapping.py — FIXED FOR ALL 3 STAGES
Root cause: Fixed thresholds (0.6 / 1.4) pushed all 68 samples into Stage 2 and 3,
            giving Stage 1 zero samples.

Fix: Use data-driven percentile thresholds (33rd / 66th percentile of the severity
     score) so each stage always gets roughly 1/3 of the samples regardless of
     the actual value distribution in your .mat files.
"""

import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import joblib
from scipy.stats import entropy as sp_entropy

DATA_PATH = "tapping_data"

# ------------------------------------------------------------------ #
#  FEATURE EXTRACTION (unchanged — rich 11-feature set per finger)
# ------------------------------------------------------------------ #
def extract_features(signal, fs=100):
    mean_val  = np.mean(signal)
    std_val   = np.std(signal)
    max_val   = np.max(signal)
    min_val   = np.min(signal)
    var_val   = np.var(signal)
    rms_val   = np.sqrt(np.mean(signal**2))

    n = len(signal)
    freqs   = np.fft.rfftfreq(n, d=1.0/fs)
    fft_mag = np.abs(np.fft.rfft(signal))

    dom_freq_idx  = np.argmax(fft_mag[1:]) + 1
    dom_freq      = freqs[dom_freq_idx]

    tap_band_mask = (freqs >= 1.0) & (freqs <= 6.0)
    tap_band_power = np.sum(fft_mag[tap_band_mask]**2)

    psd      = fft_mag**2
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spec_entropy = sp_entropy(psd_norm + 1e-10)

    peaks, _ = scipy.signal.find_peaks(signal, height=mean_val, distance=int(fs*0.15))
    if len(peaks) > 1:
        iti     = np.diff(peaks) / fs
        iti_mean = np.mean(iti)
        iti_cv   = np.std(iti) / (np.mean(iti) + 1e-10)
    else:
        iti_mean = 0.0
        iti_cv   = 1.0

    return [mean_val, std_val, max_val, min_val, var_val, rms_val,
            dom_freq, tap_band_power, spec_entropy, iti_mean, iti_cv]


# ------------------------------------------------------------------ #
#  PROCESS ALL .mat FILES — collect raw features first
# ------------------------------------------------------------------ #
rows = []

for file in sorted(os.listdir(DATA_PATH)):
    if not file.endswith(".mat"):
        continue
    file_path = os.path.join(DATA_PATH, file)
    try:
        data = scipy.io.loadmat(file_path)

        thumb_x = data["gyroThumbX"].flatten()
        thumb_y = data["gyroThumbY"].flatten()
        thumb_z = data["gyroThumbZ"].flatten()
        index_x = data["gyroIndexX"].flatten()
        index_y = data["gyroIndexY"].flatten()
        index_z = data["gyroIndexZ"].flatten()

        thumb_mag = np.sqrt(thumb_x**2 + thumb_y**2 + thumb_z**2)
        index_mag = np.sqrt(index_x**2 + index_y**2 + index_z**2)

        thumb_feats = extract_features(thumb_mag)
        index_feats = extract_features(index_mag)

        rows.append(thumb_feats + index_feats)

    except KeyError as e:
        print(f"Missing key in {file}: {e} — skipping")
    except Exception as e:
        print(f"Error in {file}: {e} — skipping")

if len(rows) == 0:
    raise RuntimeError("No .mat files processed successfully. Check DATA_PATH.")

# ------------------------------------------------------------------ #
#  BUILD DATAFRAME
# ------------------------------------------------------------------ #
FEATURE_NAMES = [
    "thumb_mean", "thumb_std", "thumb_max", "thumb_min", "thumb_var", "thumb_rms",
    "thumb_dom_freq", "thumb_tap_power", "thumb_spec_entropy",
    "thumb_iti_mean", "thumb_iti_cv",
    "index_mean", "index_std", "index_max", "index_min", "index_var", "index_rms",
    "index_dom_freq", "index_tap_power", "index_spec_entropy",
    "index_iti_mean", "index_iti_cv",
]

df = pd.DataFrame(rows, columns=FEATURE_NAMES)

if df.isnull().any().any():
    print("Warning: NaN values found — filling with column medians")
    df = df.fillna(df.median())

# ------------------------------------------------------------------ #
#  FIX: PERCENTILE-BASED STAGE ASSIGNMENT
#  Compute a combined severity score, then split into thirds.
#  This GUARANTEES all 3 stages get samples no matter what the
#  raw values look like in your specific .mat files.
# ------------------------------------------------------------------ #
severity_score = (
    df["thumb_std"]   * 0.20 +
    df["index_std"]   * 0.20 +
    df["thumb_iti_cv"] * 0.20 +
    df["index_iti_cv"] * 0.20 +
    (1.0 / (df["thumb_dom_freq"] + 0.5)) * 0.10 +
    (1.0 / (df["index_dom_freq"] + 0.5)) * 0.10
)

# Data-driven percentile thresholds — always produces 3 non-empty stages
p33 = severity_score.quantile(0.33)
p66 = severity_score.quantile(0.66)

print(f"\nSeverity score stats:")
print(f"  min={severity_score.min():.4f}  p33={p33:.4f}  p66={p66:.4f}  max={severity_score.max():.4f}")

def assign_stage(score):
    if score < p33:  return 1   # Mild
    elif score < p66: return 2  # Moderate
    else:             return 3  # Severe

df["stage"] = severity_score.apply(assign_stage)

print(f"\nStage distribution (percentile-based):")
print(df["stage"].value_counts().sort_index())

# ------------------------------------------------------------------ #
#  SAVE
# ------------------------------------------------------------------ #
df.to_csv("tapping_data.csv", index=False)

tap_feature_cols = FEATURE_NAMES   # everything except 'stage'
joblib.dump(tap_feature_cols, "tap_features.pkl")

print(f"\n✅ Tapping CSV created with ALL 3 STAGES!")
print(f"   Total samples : {len(df)}")
print(f"   Features      : {len(tap_feature_cols)}")