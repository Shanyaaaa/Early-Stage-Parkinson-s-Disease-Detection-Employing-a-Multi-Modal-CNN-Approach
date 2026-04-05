import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------------- LOAD DATA ----------------
df = pd.read_csv("parkinsons_data.csv", header=1)

# Clean column names
df.columns = df.columns.str.strip()

print("Columns Loaded:", df.columns.tolist())

# ---------------- FIND TARGET COLUMN ----------------
target_col = [col for col in df.columns if "Hoehn" in col][0]

print("✅ Using target column:", target_col)

# ---------------- CLEAN TARGET ----------------
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

# Remove rows without stage (HC & RBD)
df = df.dropna(subset=[target_col])

# ---------------- CREATE LABELS ----------------
df["status"] = df[target_col].apply(lambda x: 1 if x > 0 else 0)

# ---------------- FEATURES ----------------
X = df.drop([target_col, "status"], axis=1)

# Keep only numeric columns
X = X.select_dtypes(include=['number'])

y_status = df["status"]
y_stage = df[target_col]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_status, test_size=0.2, random_state=42
)

# ---------------- MODELS ----------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X, y_stage)

# ---------------- SAVE ----------------
joblib.dump(clf, "parkinson_classifier.pkl")
joblib.dump(reg, "parkinson_stage_model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")  # important!

print("\n✅ Models trained successfully!")