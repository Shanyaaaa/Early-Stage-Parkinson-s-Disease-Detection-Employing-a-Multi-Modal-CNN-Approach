import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# ---------------- LOAD DATA ----------------
df = pd.read_csv("parkinsons_data.csv", header=1)
df.columns = df.columns.str.strip()

# ---------------- RENAME COLUMNS ----------------
df = df.rename(columns={
    "Age (years)": "Age",
    "UPDRS III total (-)": "UPDRS III",
    "Duration of disease from first symptoms (years)": "Disease Duration",
    "18. Speech": "Speech Score",
    "19. Facial Expression": "Facial Expression",
    "20. Tremor at Rest - head": "Tremor (Head)",
    "Levodopa equivalent (mg/day)": "Levodopa (mg/day)"
})

# ---------------- TARGET ----------------
target_col = "Hoehn & Yahr scale (-)"

df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

# ✅ IMPORTANT → include healthy patients
df[target_col] = df[target_col].fillna(0)

# ---------------- FEATURES ----------------
features = [
    "Age",
    "UPDRS III",
    "Disease Duration",
    "Speech Score",
    "Facial Expression",
    "Tremor (Head)",
    "Levodopa (mg/day)"
]

# Clean feature values
X = df[features].replace("-", pd.NA)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

y = df[target_col]

# ---------------- MODEL ----------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X, y)

# ---------------- SAVE ----------------
joblib.dump(model, "parkinson_model.pkl")
joblib.dump(features, "features.pkl")

print("✅ Model trained successfully (Stages 0–5)")