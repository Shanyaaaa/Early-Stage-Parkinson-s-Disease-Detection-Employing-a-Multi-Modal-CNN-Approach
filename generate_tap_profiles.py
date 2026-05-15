"""
generate_tap_profiles.py
Run ONCE after training the tap model.
Creates tap_stage_profiles.pkl — mean feature vector per stage.
"""
import pandas as pd
import numpy as np
import joblib

df           = pd.read_csv("tapping_data.csv")
tap_features = joblib.load("tap_features.pkl")
X, y         = df[tap_features], df["stage"] - 1   # 0-indexed

profiles = {}
for s in [0, 1, 2]:
    mask = y == s
    profiles[s] = X[mask].mean().values if mask.sum() > 0 else X.mean().values
    print(f"Stage {s+1}: {mask.sum()} samples, profile[0:3]={profiles[s][:3].round(4)}")

joblib.dump(profiles, "tap_stage_profiles.pkl")
print("\n✅ tap_stage_profiles.pkl saved!")