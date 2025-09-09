import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

print("[Step 1] Loading and cleaning files...")

def load_and_clean(path, rename_col=None):
    df = pd.read_csv(path)
    if rename_col:
        df.rename(columns=rename_col, inplace=True)
    df['address'] = df['address'].astype(str).str.lower().str.strip()
    return df

# Load files
df_retention = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/retention_AND.csv")[['address', 'retained_25']]
df_retention['retained_25'] = df_retention['retained_25'].astype(int)

df_com = load_and_clean("/home/s3986160/master-thesis/Lastround /output/community_users_with_all_centralities.csv")[['address', 'community_id', 'community_size', 'degree', 'betweenness', 'clustering', 'weighted_degree']]
df_degree = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/raw_degree_2020.csv", rename_col={'user': 'address'})
df_weighted = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/weighted_degree_2020.csv", rename_col={'user': 'address'})
df_clustering = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/clustering_coefficients_2020.csv")
df_betweenness = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/betweenness_centrality_2020.csv")
df_cyclic = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/cyclic_vs_retention_25.csv")[['address', 'cyclic_status']]
df_cyclic['cyclic_status'] = df_cyclic['cyclic_status'].map({'acyclic': 0, 'cyclic': 1})

print("[Step 2] Merging all features...")
df = df_retention.merge(df_com, on='address', how='inner')
df = df.merge(df_degree, on='address', how='left')
df = df.merge(df_weighted, on='address', how='left')
df = df.merge(df_clustering, on='address', how='left')
df = df.merge(df_betweenness, on='address', how='left')
df = df.merge(df_cyclic, on='address', how='left')

print("[Step 3] Handling missing values...")
df.fillna(0, inplace=True)
print(f"[DEBUG] Final merged dataset shape: {df.shape}")

print("[Step 4] Encoding community_id...")
df['community_id'] = df['community_id'].astype(str)
df = pd.get_dummies(df, columns=['community_id'], drop_first=True)

X = df.drop(columns=['address', 'retained_25'])
y = df['retained_25'].astype(int)

print("[Step 5] Splitting into train/val/test...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("[Step 6] Applying SMOTE to training set...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_res.shape}, Positives: {sum(y_train_res)}")

print("[Step 7] Tuning Random Forest with 5-fold CV...")
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
    scoring='recall',  # Optimizing for recall
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train_res, y_train_res)
print("[Best Parameters]:", search.best_params_)

# ---- EVALUATION FUNCTION ----
def evaluate_model(X, y, proba, label=""):
    preds_default = (proba >= 0.5).astype(int)
    preds_adjusted = (proba >= 0.3).astype(int)

    print(f"\n📊 Evaluation ({label}) — Default Threshold (0.5)")
    print(classification_report(y, preds_default))
    print("ROC AUC:", roc_auc_score(y, proba))

    print(f"\n📊 Evaluation ({label}) — Adjusted Threshold (0.3)")
    print(classification_report(y, preds_adjusted))
    print("ROC AUC:", roc_auc_score(y, proba))

print("[Step 8] Evaluation on validation set...")
val_proba = search.predict_proba(X_val)[:, 1]
evaluate_model(X_val, y_val, val_proba, "Validation")

print("[Step 9] Evaluation on test set...")
test_proba = search.predict_proba(X_test)[:, 1]
evaluate_model(X_test, y_test, test_proba, "Test")

print("[Step 10] Plotting feature importance...")
importances = search.best_estimator_.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values[:25], y=feat_imp.index[:25], palette="viridis")
plt.title("Top 25 Feature Importances (Global + Community)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_global_community_p03.png", dpi=300)
plt.show()

print("✅ Model (Global + Community) with p=0.3 and recall optimization completed.")
