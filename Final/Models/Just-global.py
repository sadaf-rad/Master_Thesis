import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE

print("[Step 1] Loading data...")

# Load datasets
df_retention = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/retention_AND.csv")
df_degree = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/raw_degree_2020.csv")
df_weighted = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/weighted_degree_2020.csv")
df_clustering = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/clustering_coefficients_2020.csv")
df_betweenness = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/betweenness_centrality_2020.csv")
df_cyclic = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/cyclic_vs_retention_25.csv")

# Clean & rename
df_degree.rename(columns={"user": "address"}, inplace=True)
df_weighted.rename(columns={"user": "address"}, inplace=True)

print("[Step 2] Merging features...")
df = df_retention[['address', 'retained_25']].copy()
df = df.merge(df_degree, on='address', how='left')
df = df.merge(df_weighted, on='address', how='left')
df = df.merge(df_clustering, on='address', how='left')
df = df.merge(df_betweenness, on='address', how='left')
df = df.merge(df_cyclic[['address', 'cyclic_status']], on='address', how='left')

df.dropna(inplace=True)
df['cyclic_status'] = df['cyclic_status'].map({'cyclic': 1, 'acyclic': 0})

X = df.drop(columns=['address', 'retained_25'])
y = df['retained_25']

print("[Step 3] Splitting train/val/test...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("[Step 4] Applying SMOTE oversampling to training set...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_res.shape}, Positive samples: {np.sum(y_train_res)}")

print("[Step 5] Tuning Random Forest with class_weight='balanced' and 5-fold CV...")

param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

model = RandomForestClassifier(class_weight="balanced", random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=10,
    scoring='recall',
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train_res, y_train_res)
print("[Best Params]:", search.best_params_)

print("[Step 6] Validation performance (Threshold = 0.3):")
val_proba = search.predict_proba(X_val)[:, 1]
val_preds_custom = (val_proba >= 0.3).astype(int)
print(classification_report(y_val, val_preds_custom))
print("ROC AUC (val):", roc_auc_score(y_val, val_proba))
print("Recall (val, threshold 0.3):", recall_score(y_val, val_preds_custom))

print("[Step 7] Test performance (Threshold = 0.3):")
test_proba = search.predict_proba(X_test)[:, 1]
test_preds_custom = (test_proba >= 0.3).astype(int)
print(classification_report(y_test, test_preds_custom))
print("ROC AUC (test):", roc_auc_score(y_test, test_proba))
print("Recall (test, threshold 0.3):", recall_score(y_test, test_preds_custom))

# === Feature importance ===
print("[Step 8] Saving feature importance plot...")
importances = search.best_estimator_.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
plt.title("Feature Importance (Global Features)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_rf_recall_p03.png", dpi=300)
plt.show()

print("✅ Model training, tuning, and evaluation complete.")