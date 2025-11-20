# =============================================================================
# TrainStateModel_WESAD_rel.py
# =============================================================================
#
# PURPOSE
# -------
# Train a machine-learning model that predicts *user state* (baseline / stress /
# relax, etc.) from baseline-normalized physiological features extracted from the
# WESAD dataset.
#
# DATA PIPELINE
# -------------
#   1. preprocess_wesad_rel.py
#        → creates Data/wesad_features_rel.csv
#        → each row = one time window for one subject
#        → columns:
#             - subject, label, timestamp
#             - hr_mean, hrv_rmssd, eda_mean, eda_std, eda_deriv_std
#             - hr_mean_rel, hrv_rmssd_rel, eda_mean_rel, eda_std_rel, eda_deriv_std_rel
#
#   2. THIS SCRIPT
#        - loads that CSV
#        - uses only *_rel features (relative to subject baseline)
#        - builds a RandomForest model inside an sklearn Pipeline
#        - uses GroupKFold cross-validation (grouped by subject)
#        - trains on all data
#        - saves the trained model to Models/user_state_model.pkl
#
# WHY GROUPKFold?
# ---------------
# We want to simulate "new subjects" at test time, so we must ensure that:
#   - training folds and validation folds consist of *different subjects*.
# GroupKFold does exactly that: each fold contains different subject IDs.
#
# WHY RELATIVE FEATURES?
# ----------------------
# Because each subject has their own physiological baseline. By subtracting
# per-subject baseline means, we get features that represent "how far from
# baseline" the subject is, which often generalizes better to new people.
#
# =============================================================================

import os

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import joblib


# =============================================================================
# 1. LOAD PREPROCESSED WESAD FEATURES
# =============================================================================

# Relative-feature CSV produced by preprocess_wesad_rel.py
csv_path = "Data/wesad_features_rel.csv"

# Read the entire dataset into a pandas DataFrame.
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from {csv_path}")

# Feature columns we will use for training.
# NOTE: these are all *relative* features (baseline-corrected).
rel_features = [
    "hr_mean_rel",
    "hrv_rmssd_rel",
    "eda_mean_rel",
    "eda_std_rel",
    "eda_deriv_std_rel"
]

# X: 2D numpy array of feature values (n_samples x n_features)
X = df[rel_features].values

# y: target labels (e.g., "baseline", "stress", "relax")
y = df["label"].values

# groups: subject IDs; used to ensure that train/test splits separate subjects.
groups = df["subject"].values


# =============================================================================
# 2. HANDLE MISSING VALUES
# =============================================================================

# Some windows might have missing HRV or HR (e.g., not enough peaks).
# These appear as NaN in X. We replace them with 0.0.
#
# Why 0.0?
# - Because these are *relative* features: 0.0 means "equal to baseline".
# - A missing HRV value is then treated as "no deviation from baseline".
X = np.nan_to_num(X, nan=0.0)


# =============================================================================
# 3. CLASS WEIGHTS (HANDLE CLASS IMBALANCE)
# =============================================================================

# Different states (baseline, stress, relax) usually appear with different
# frequencies in WESAD. If we just train naively, the model might learn
# "always predict baseline" and still get high accuracy.
#
# To counter this, we compute balanced class weights:
#   class_weight[c] is inversely proportional to how often 'c' appears in y.
classes = np.unique(y)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weight = dict(zip(classes, weights))

print("Class weights (label → weight):")
for lbl, w in class_weight.items():
    print(f"  {lbl}: {w:.3f}")


# =============================================================================
# 4. DEFINE MODEL PIPELINE
# =============================================================================

# We use an sklearn Pipeline, which is a chain of processing steps:
#
#   Step 1: StandardScaler
#       - Standardizes each feature to mean ~0 and standard deviation ~1
#       - Important because trees in RandomForest can benefit from more
#         homogeneous feature scales, and it’s generally good practice.
#
#   Step 2: RandomForestClassifier
#       - Ensemble of decision trees
#       - Handles non-linear relationships and mixed feature types well
#       - Robust to outliers and often works well out-of-the-box
#
# Parameters for RandomForest:
#   - n_estimators=400: number of trees in the forest
#   - max_depth=10: limit tree depth → avoids overfitting
#   - random_state=42: reproducible results
#   - class_weight=class_weight: incorporate the balanced weights computed above
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        random_state=42,
        class_weight=class_weight
    ))
])


# =============================================================================
# 5. SUBJECT-WISE CROSS-VALIDATION (GroupKFold)
# =============================================================================

# GroupKFold with 5 splits:
#   - On each split, some subjects are used for training, others for validation.
#   - A subject never appears in both train and validation in the same fold.
gkf = GroupKFold(n_splits=5)

# cross_val_score will:
#   - iterate through the folds
#   - fit the model on the training subjects
#   - compute accuracy on validation subjects
scores = cross_val_score(
    model,
    X,
    y,
    cv=gkf.split(X, y, groups=groups)
)

print(f"\nGroupKFold subject-wise accuracy: "
      f"{scores.mean():.3f} ± {scores.std():.3f}")


# =============================================================================
# 6. TRAIN ON ALL DATA AND SAVE MODEL
# =============================================================================

# After we have an idea of performance, we fit the model on all available data.
model.fit(X, y)

# Ensure the Models/ directory exists
os.makedirs("Models", exist_ok=True)

# Save the entire pipeline (scaler + RandomForest) as a single object.
out_path = "Models/user_state_model.pkl"
joblib.dump(model, out_path)

print(f"\n✅ Trained model saved to {out_path}")


# =============================================================================
# 7. SANITY CHECK: FIT/PREDICT ON FULL DATA (NOT REAL EVALUATION)
# =============================================================================

# WARNING:
#   The following is only a quick *diagnostic* check.
#   We train and test on the same data, so the scores here are optimistic.
#   The real generalization ability is captured by the GroupKFold accuracy above.
y_pred = model.predict(X)

print("\nSanity-check report (train = test = all data):\n")
print(classification_report(y, y_pred))

print("Confusion matrix (rows = true labels, cols = predicted labels):\n")
print(confusion_matrix(y, y_pred))
