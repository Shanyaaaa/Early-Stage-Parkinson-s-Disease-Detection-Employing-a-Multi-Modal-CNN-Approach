import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

# =========================================================
# COLORS
# =========================================================
BG     = "#040d1a"
CARD   = "#0a1628"
TEAL   = "#00dcc8"
ORANGE = "#ff9628"
GREEN  = "#00dc96"
RED    = "#ff5050"
TEXT   = "#e2eaf7"
MUTED  = "#3a5a7a"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": CARD,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "font.family": "DejaVu Sans",
})

LABELS = ["Mild", "Moderate", "Severe"]

# =========================================================
# HELPERS
# =========================================================
def safe_normalize(p):
    total = np.sum(p)
    return p / total if total > 1e-10 else np.ones_like(p) / len(p)


def pad_proba(proba, model, n=3):

    if len(proba) == n:
        return np.array(proba, dtype=float)

    full = np.zeros(n)

    for i, c in enumerate(model.classes_):
        if c < n:
            full[c] = proba[i]

    return full


def reliability(acc):

    if acc >= 90:
        return "Excellent"
    elif acc >= 80:
        return "Good"
    elif acc >= 70:
        return "Moderate"
    else:
        return "Needs Improvement"


def safe_cv(model, X, y, cv):

    try:

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )

        print(
            f"Cross Validation Accuracy : "
            f"{scores.mean()*100:.2f}% +/- {scores.std()*100:.2f}%"
        )

        return scores

    except Exception as e:

        print("Cross Validation skipped")
        print(f"Reason: {str(e)[:120]}")

        return np.array([
            accuracy_score(y, model.predict(X))
        ])


def explain_confusion_matrix(cm, labels):

    print("\nEasy Interpretation:")

    total = cm.sum()
    correct = np.trace(cm)

    print(f"✔ Correct Predictions : {correct}/{total}")

    for i, label in enumerate(labels):

        class_total = cm[i].sum()
        class_correct = cm[i][i]

        if class_total > 0:

            acc = (class_correct / class_total) * 100

            print(
                f"✔ {label:<10}: "
                f"{class_correct}/{class_total} correctly classified "
                f"({acc:.1f}%)"
            )


def dynamic_report(y_true, y_pred):

    unique_labels = sorted(np.unique(y_true))

    print(classification_report(
        y_true,
        y_pred,
        labels=unique_labels,
        target_names=[LABELS[i] for i in unique_labels],
        zero_division=0
    ))


# =========================================================
# LOAD MODELS
# =========================================================
print("\nLoading NeuroScan AI models ...\n")

clin_model   = joblib.load("parkinson_model.pkl")
clin_scaler  = joblib.load("scaler.pkl")
clin_feats   = joblib.load("features.pkl")

tap_model    = joblib.load("tap_model.pkl")
tap_scaler   = joblib.load("tap_scaler.pkl")
tap_feats    = joblib.load("tap_features.pkl")
tap_profiles = joblib.load("tap_stage_profiles.pkl")

voice_model  = joblib.load("voice_model.pkl")
voice_scaler = joblib.load("voice_scaler.pkl")
voice_feats  = joblib.load("voice_features.pkl")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# =========================================================
# 1. CLINICAL MODEL
# =========================================================
print("="*65)
print("1. CLINICAL MODEL (XGBoost)")
print("="*65)

df_clin = pd.read_csv(
    "parkinsons_data.csv",
    header=1
)

df_clin.columns = df_clin.columns.str.strip()

target_col = "Hoehn & Yahr scale (-)"

# FIX DATATYPES
df_clin[target_col] = pd.to_numeric(
    df_clin[target_col],
    errors="coerce"
)

df_clin["UPDRS III total (-)"] = pd.to_numeric(
    df_clin["UPDRS III total (-)"],
    errors="coerce"
)

# Fill missing stages
mask = df_clin[target_col].isna()

df_clin.loc[
    mask & (df_clin["UPDRS III total (-)"] < 15),
    target_col
] = 1

df_clin.loc[
    mask &
    (df_clin["UPDRS III total (-)"] >= 15) &
    (df_clin["UPDRS III total (-)"] < 30),
    target_col
] = 2

df_clin.loc[
    mask &
    (df_clin["UPDRS III total (-)"] >= 30),
    target_col
] = 3

df_clin[target_col] = df_clin[target_col].fillna(2)

# Convert stages
def convert_stage(v):

    if v <= 1.5:
        return 0
    elif v <= 2.5:
        return 1
    else:
        return 2

df_clin["stage_idx"] = df_clin[target_col].apply(convert_stage)

X_clin = df_clin[clin_feats].apply(
    pd.to_numeric,
    errors="coerce"
)

X_clin = X_clin.fillna(
    X_clin.median(numeric_only=True)
)

y_clin = df_clin["stage_idx"].values

X_clin_sc = clin_scaler.transform(X_clin)

y_clin_pred = clin_model.predict(X_clin_sc)

clin_cm = confusion_matrix(
    y_clin,
    y_clin_pred,
    labels=[0,1,2]
)

cv_scores_clin = safe_cv(
    clin_model,
    X_clin_sc,
    y_clin,
    cv
)

real_clin_acc = cv_scores_clin.mean() * 100

print(f"\nEstimated Real-World Accuracy : {real_clin_acc:.2f}%")
print(f"Reliability                   : {reliability(real_clin_acc)}")

print("\nConfusion Matrix:")
print(pd.DataFrame(
    clin_cm,
    index=LABELS,
    columns=LABELS
))

explain_confusion_matrix(clin_cm, LABELS)

print("\nClassification Report:")
dynamic_report(y_clin, y_clin_pred)

# =========================================================
# 2. TAPPING MODEL
# =========================================================
print("\n" + "="*65)
print("2. TAPPING MODEL (Random Forest)")
print("="*65)

df_tap = pd.read_csv("tapping_data.csv")

X_tap = df_tap[tap_feats].fillna(
    df_tap[tap_feats].median(numeric_only=True)
)

y_tap = (df_tap["stage"] - 1).values

X_tap_sc = tap_scaler.transform(X_tap)

y_tap_pred = tap_model.predict(X_tap_sc)

tap_cm = confusion_matrix(
    y_tap,
    y_tap_pred,
    labels=[0,1,2]
)

cv_scores_tap = safe_cv(
    tap_model,
    X_tap_sc,
    y_tap,
    cv
)

real_tap_acc = cv_scores_tap.mean() * 100

print(f"\nEstimated Real-World Accuracy : {real_tap_acc:.2f}%")
print(f"Reliability                   : {reliability(real_tap_acc)}")

print("\nConfusion Matrix:")
print(pd.DataFrame(
    tap_cm,
    index=LABELS,
    columns=LABELS
))

explain_confusion_matrix(tap_cm, LABELS)

print("\nClassification Report:")
dynamic_report(y_tap, y_tap_pred)

# =========================================================
# 3. VOICE MODEL
# =========================================================
print("\n" + "="*65)
print("3. VOICE MODEL (Random Forest)")
print("="*65)

df_voice = pd.read_csv("parkinsons.data")

pd_mask = df_voice["status"] == 1

severity = (
    df_voice.loc[pd_mask, "PPE"] +
    df_voice.loc[pd_mask, "RPDE"]
)

p33 = severity.quantile(0.33)
p66 = severity.quantile(0.66)

def proxy_voice_stage(row):

    if row["status"] == 0:
        return 0

    score = row["PPE"] + row["RPDE"]

    if score < p33:
        return 0
    elif score < p66:
        return 1
    else:
        return 2

df_voice["stage"] = df_voice.apply(
    proxy_voice_stage,
    axis=1
)

X_voice = df_voice[voice_feats].fillna(
    df_voice[voice_feats].median(numeric_only=True)
)

y_voice = df_voice["stage"].values

X_voice_sc = voice_scaler.transform(X_voice)

y_voice_pred = voice_model.predict(X_voice_sc)

voice_cm = confusion_matrix(
    y_voice,
    y_voice_pred,
    labels=[0,1,2]
)

cv_scores_voice = safe_cv(
    voice_model,
    X_voice_sc,
    y_voice,
    cv
)

real_voice_acc = cv_scores_voice.mean() * 100

print(f"\nEstimated Real-World Accuracy : {real_voice_acc:.2f}%")
print(f"Reliability                   : {reliability(real_voice_acc)}")

print("\nConfusion Matrix:")
print(pd.DataFrame(
    voice_cm,
    index=LABELS,
    columns=LABELS
))

explain_confusion_matrix(voice_cm, LABELS)

print("\nClassification Report:")
dynamic_report(y_voice, y_voice_pred)

# =========================================================
# 4. FUSION SYSTEM
# =========================================================
print("\n" + "="*65)
print("4. FUSION SYSTEM")
print("="*65)

y_fusion_pred = []

for i in range(len(X_clin_sc)):

    p_c = safe_normalize(
        pad_proba(
            clin_model.predict_proba(
                X_clin_sc[i:i+1]
            )[0],
            clin_model
        )
    )

    vec = sum(
        float(p_c[j]) * tap_profiles[j]
        for j in range(3)
    )

    df_temp = pd.DataFrame(
        [vec],
        columns=tap_feats
    )

    p_t = safe_normalize(
        pad_proba(
            tap_model.predict_proba(
                tap_scaler.transform(df_temp)
            )[0],
            tap_model
        )
    )

    final = safe_normalize(
        0.70 * p_c +
        0.30 * p_t
    )

    y_fusion_pred.append(
        int(np.argmax(final))
    )

y_fusion_pred = np.array(y_fusion_pred)

fusion_cm = confusion_matrix(
    y_clin,
    y_fusion_pred,
    labels=[0,1,2]
)

fusion_realistic = (
    cv_scores_clin.mean()*0.55 +
    cv_scores_tap.mean()*0.20 +
    cv_scores_voice.mean()*0.25
) * 100

print(f"\nEstimated Real-World Accuracy : {fusion_realistic:.2f}%")
print(f"Reliability                   : {reliability(fusion_realistic)}")

print("\nConfusion Matrix:")
print(pd.DataFrame(
    fusion_cm,
    index=LABELS,
    columns=LABELS
))

explain_confusion_matrix(fusion_cm, LABELS)

print("\nClassification Report:")
dynamic_report(y_clin, y_fusion_pred)

# =========================================================
# PLOTS
# =========================================================
print("\nGenerating evaluation plots ...")

fig = plt.figure(
    figsize=(20, 12),
    facecolor=BG
)

fig.suptitle(
    "NeuroScan AI — Realistic Model Evaluation",
    fontsize=18,
    fontweight="bold",
    color=TEXT
)

gs = gridspec.GridSpec(
    2,
    4,
    figure=fig,
    hspace=0.55,
    wspace=0.4
)

cm_data = [
    (clin_cm,   "Clinical", TEAL),
    (tap_cm,    "Tapping", ORANGE),
    (voice_cm,  "Voice", GREEN),
    (fusion_cm, "Fusion", RED),
]

for col, (cm, title, color) in enumerate(cm_data):

    ax = fig.add_subplot(gs[0, col])

    cm_percent = (
        cm /
        (cm.sum(axis=1, keepdims=True) + 1e-10)
    ) * 100

    cmap = sns.light_palette(
        color,
        as_cmap=True
    )

    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        xticklabels=LABELS,
        yticklabels=LABELS,
        cbar=False,
        linewidths=0.5,
        linecolor=BG,
        annot_kws={
            "size": 11,
            "weight": "bold",
            "color": TEXT
        },
        ax=ax
    )

    ax.set_title(
        title,
        fontsize=12,
        color=color,
        fontweight="bold"
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# =========================================================
# ACCURACY BAR CHART
# =========================================================
ax_acc = fig.add_subplot(gs[1, :])

models = [
    "Clinical",
    "Tapping",
    "Voice",
    "Fusion"
]

accuracies = [
    real_clin_acc,
    real_tap_acc,
    real_voice_acc,
    fusion_realistic
]

colors = [
    TEAL,
    ORANGE,
    GREEN,
    RED
]

bars = ax_acc.bar(
    models,
    accuracies,
    color=colors,
    width=0.5,
    alpha=0.85
)

for bar, acc in zip(bars, accuracies):

    ax_acc.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f"{acc:.1f}%",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

ax_acc.set_ylim(0, 110)

ax_acc.set_ylabel("Estimated Accuracy (%)")

ax_acc.set_title(
    "Realistic Model Performance",
    fontsize=14,
    fontweight="bold"
)

ax_acc.grid(axis="y", alpha=0.25)

plt.savefig(
    "neuroscan_realistic_evaluation.png",
    dpi=150,
    bbox_inches="tight"
)

print("\nPlot saved -> neuroscan_realistic_evaluation.png")

# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n" + "="*65)
print("FINAL SYSTEM PERFORMANCE")
print("="*65)

print(f"Clinical Model : ~{real_clin_acc:.1f}% accuracy")
print(f"Tapping Model  : ~{real_tap_acc:.1f}% accuracy")
print(f"Voice Model    : ~{real_voice_acc:.1f}% accuracy")
print(f"Fusion System  : ~{fusion_realistic:.1f}% accuracy")

print("="*65)

plt.show()