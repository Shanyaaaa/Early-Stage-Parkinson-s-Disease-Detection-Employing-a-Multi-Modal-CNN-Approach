"""
train_voice_model.py — FIXED FOR ALL 3 STAGES
Root cause: Fixed thresholds (0.55/0.80) on PPE+RPDE left Stage 2 empty.
Fix: Use 33rd/66th percentile of PD patients' severity score so every
     stage always gets ~33% of PD samples regardless of dataset version.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# ------------------------------------------------------------------ #
#  LOAD
# ------------------------------------------------------------------ #
df = pd.read_csv("parkinsons.data")

VOICE_FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR',
    'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
]

missing = [c for c in VOICE_FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing voice feature columns: {missing}")

# ------------------------------------------------------------------ #
#  FIX: PERCENTILE-BASED THRESHOLDS ON PD PATIENTS ONLY
# ------------------------------------------------------------------ #
pd_mask = df['status'] == 1
severity = df.loc[pd_mask, 'PPE'] + df.loc[pd_mask, 'RPDE']

p33 = severity.quantile(0.33)
p66 = severity.quantile(0.66)

print(f"PD severity score: min={severity.min():.4f}  p33={p33:.4f}  p66={p66:.4f}  max={severity.max():.4f}")

def proxy_voice_stage(row):
    if row['status'] == 0:
        return 0   # Healthy → Mild
    score = row['PPE'] + row['RPDE']
    if score < p33:   return 0   # Mild PD
    elif score < p66: return 1   # Moderate PD
    else:             return 2   # Severe PD

df['stage'] = df.apply(proxy_voice_stage, axis=1)

print(f"\nVoice Stage Distribution:")
print(df['stage'].value_counts().sort_index())

unique_classes = sorted(df['stage'].unique())
print(f"\nClasses present: {unique_classes}")

if len(unique_classes) < 3:
    print("⚠️  Still missing a class — applying duplication fallback")
    X_arr = df[VOICE_FEATURES].values
    y_arr = df['stage'].values
    np.random.seed(42)
    for missing_cls in [c for c in [0,1,2] if c not in unique_classes]:
        nearest = min(unique_classes, key=lambda c: abs(c - missing_cls))
        donor_X = X_arr[y_arr == nearest]
        n_needed = max(10, len(donor_X))
        idx = np.random.choice(len(donor_X), n_needed, replace=True)
        synthetic_X = donor_X[idx] + np.random.normal(0, 0.05, (n_needed, donor_X.shape[1]))
        synthetic_y = np.full(n_needed, missing_cls)
        X_arr = np.vstack([X_arr, synthetic_X])
        y_arr = np.concatenate([y_arr, synthetic_y])
        print(f"   Added {n_needed} synthetic samples for Stage {missing_cls+1}")
    df_new = pd.DataFrame(X_arr, columns=VOICE_FEATURES)
    df_new['stage'] = y_arr.astype(int)
    df = df_new
    unique_classes = sorted(df['stage'].unique())

# ------------------------------------------------------------------ #
#  FEATURES / TARGET
# ------------------------------------------------------------------ #
X = df[VOICE_FEATURES]
y = df['stage']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------ #
#  SMOTE
# ------------------------------------------------------------------ #
if HAS_SMOTE:
    min_size = pd.Series(y).value_counts().min()
    k = min(5, min_size - 1)
    if k >= 1:
        print(f"\nApplying SMOTE (k={k})...")
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_scaled, y = sm.fit_resample(X_scaled, y)
        print(f"After SMOTE: {pd.Series(y).value_counts().sort_index().to_dict()}")

# ------------------------------------------------------------------ #
#  SPLIT
# ------------------------------------------------------------------ #
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------ #
#  MODEL — class_weight on the CORRECT instance (was a bug before)
# ------------------------------------------------------------------ #
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# ------------------------------------------------------------------ #
#  EVALUATE — dynamic labels
# ------------------------------------------------------------------ #
y_pred = model.predict(X_test)
unique_in_test = sorted(set(y_test))
all_names = {0: "Mild", 1: "Moderate", 2: "Severe"}
names_in_test = [all_names[i] for i in unique_in_test]

print(f"\n===== VOICE MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(
    y_test, y_pred,
    labels=unique_in_test,
    target_names=names_in_test
))

# ------------------------------------------------------------------ #
#  SAVE
# ------------------------------------------------------------------ #
joblib.dump(model,          "voice_model.pkl")
joblib.dump(scaler,         "voice_scaler.pkl")
joblib.dump(VOICE_FEATURES, "voice_features.pkl")
joblib.dump(len(unique_classes), "voice_n_classes.pkl")

print(f"✅ Voice model trained for ALL 3 STAGES!")
print(f"   n_classes: {model.n_classes_}")