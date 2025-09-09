# Full XGBoost Pipeline with Optuna Tuning, MICE Imputation, SMOTE-ENN, Threshold Optimization, and Cross-Validation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, f1_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import optuna

# === Step 1: Load and clean ===
def load_and_clean(path, rename_col=None):
    df = pd.read_csv(path)
    if rename_col:
        df.rename(columns=rename_col, inplace=True)
    if 'address' in df.columns:
        df['address'] = df['address'].astype(str).str.lower().str.strip()
    return df

print("[1] Loading files...")
df_retention = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/retention_AND.csv")[['address', 'retained_25']]
df_retention['retained_25'] = df_retention['retained_25'].astype(int)

# Global features
df_degree = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/raw_degree_2020.csv", {'user': 'address'})
df_weighted = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/weighted_degree_2020.csv", {'user': 'address'})
df_clustering = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/clustering_coefficients_2020.csv")
df_betweenness = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/betweenness_centrality_2020.csv")
df_cyclic = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/cyclic_vs_retention_25.csv")[['address', 'cyclic_status']]
df_cyclic['cyclic_status'] = df_cyclic['cyclic_status'].map({'acyclic': 0, 'cyclic': 1})

# Community-level features
df_user_comm = load_and_clean("/home/s3986160/master-thesis/Improvements/community_user_centralities.csv", {'user': 'address'})
df_user_comm.drop(columns=['retained_25'], errors='ignore', inplace=True)
df_user_comm.rename(columns={
    "degree": "com_degree",
    "weighted_degree": "com_weighted_degree",
    "clustering": "com_clustering",
    "betweenness": "com_betweenness"
}, inplace=True)
df_comm_stats = pd.read_csv("/home/s3986160/master-thesis/Improvements/centrality_stats_by_community.csv")
df_comm_stats.rename(columns={
    "degree_mean": "com_degree_mean",
    "degree_std": "com_degree_std",
    "weighted_degree_mean": "com_weighted_degree_mean",
    "weighted_degree_std": "com_weighted_degree_std",
    "clustering_mean": "com_clustering_mean",
    "clustering_std": "com_clustering_std",
    "betweenness_mean": "com_betweenness_mean",
    "betweenness_std": "com_betweenness_std"
}, inplace=True)

# Motif features
motif_counts = load_and_clean("/home/s3986160/master-thesis/Results/user_motif_details_2020.csv", {'user': 'address'})
motif_counts_pivot = motif_counts.pivot_table(index='address', columns='motif_type', values='user_count', aggfunc='sum')
motif_counts_pivot.columns = [f'motif_count_{col}' for col in motif_counts_pivot.columns]
motif_counts_pivot.reset_index(inplace=True)

motif_gaps = load_and_clean("/home/s3986160/master-thesis/Results/user_motif_timegaps_2020.csv", {'user': 'address'})
motif_gaps_pivot = motif_gaps.pivot_table(index='address', columns='motif_type', values='avg_time_gap_sec', aggfunc='mean')
motif_gaps_pivot.columns = [f'avg_gap_{col}' for col in motif_gaps_pivot.columns]
motif_gaps_pivot.reset_index(inplace=True)

# Demographics
df_demo = load_and_clean("/home/s3986160/master-thesis/Results/global_users_demographics_2020_2021.csv")
df_demo = df_demo[['address', 'gender', 'area_name', 'business_type']].fillna('Unknown')

# === Step 2: Merge ===
print("[2] Merging features...")
df = df_retention.merge(df_degree, on='address', how='left')
df = df.merge(df_weighted, on='address', how='left')
df = df.merge(df_clustering, on='address', how='left')
df = df.merge(df_betweenness, on='address', how='left')
df = df.merge(df_cyclic, on='address', how='left')
df = df.merge(df_user_comm, on='address', how='left')
df = df.merge(df_comm_stats, on='consensus_community_id', how='left')
df = df.merge(motif_counts_pivot, on='address', how='left')
df = df.merge(motif_gaps_pivot, on='address', how='left')
df = df.merge(df_demo, on='address', how='left')
df.drop(columns=["consensus_community_id"], inplace=True)

# === Step 3: Encode categoricals ===
print("[3] Encoding categorical variables...")
df = pd.get_dummies(df, columns=['gender', 'area_name', 'business_type'], drop_first=True)

# === Step 4: Split ===
print("[4] Preparing training set...")
X = df.drop(columns=['address', 'retained_25'])
y = df['retained_25']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# === Step 5: MICE ===
print("[5] Imputing with MICE...")
imputer = IterativeImputer(random_state=42)
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# === Step 6: SMOTE-ENN ===
print("[6] Applying SMOTE-ENN...")
X_train_res, y_train_res = SMOTEENN(random_state=42).fit_resample(X_train_imp, y_train)

# === Step 7: Optuna tuning ===
print("[7] Optuna tuning...")
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False
    }
    model = XGBClassifier(**params)
    model.fit(X_train_res, y_train_res)
    preds = model.predict(X_val_imp)
    return f1_score(y_val, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_params

# === Step 8: Cross-validation ===
print("[8] Cross-validation on best model...")
cv_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(cv_model, X_train_res, y_train_res, scoring='accuracy', cv=cv)
cv_f1 = cross_val_score(cv_model, X_train_res, y_train_res, scoring='f1', cv=cv)
cv_auc = cross_val_score(cv_model, X_train_res, y_train_res, scoring='roc_auc', cv=cv)
print("\nCross-Validation Results:")
print(f"Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"F1-Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print(f"ROC AUC:  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# === Step 9: Final model training ===
print("[9] Training final model...")
model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train_res, y_train_res)

# === Step 10: Threshold tuning ===
val_probs = model.predict_proba(X_val_imp)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
f1s = 2 * (precision * recall) / (precision + recall + 1e-6)
best_thresh = thresholds[np.argmax(f1s)]

val_preds = (val_probs >= best_thresh).astype(int)
print("\nValidation Report")
print(classification_report(y_val, val_preds))
print("ROC AUC:", roc_auc_score(y_val, val_probs))

# === Step 11: Final test ===
test_probs = model.predict_proba(X_test_imp)[:, 1]
test_preds = (test_probs >= best_thresh).astype(int)
print("\nTest Report")
print(classification_report(y_test, test_preds))
print("ROC AUC:", roc_auc_score(y_test, test_probs))

# === Step 12: Plot importance ===
plt.figure(figsize=(10, 6))
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feat_imp.values[:25], y=feat_imp.index[:25])
plt.title("Top 25 Feature Importances")
plt.tight_layout()
plt.savefig("improve_leiden_xgb_feature_importance_model_leiden_mice_cv.png", dpi=300)
plt.show()

print("\n✅ Training and CV complete.")
