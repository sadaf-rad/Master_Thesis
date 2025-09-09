import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_recall_curve, f1_score
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

# === Output directory ===
output_dir = "/home/s3986160/master-thesis/Retention/new definition/filtered/models"
os.makedirs(output_dir, exist_ok=True)

print("[Step 1] Loading retention labels...")
df_retention = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/retention_AND.csv")
df_retention = df_retention[['address', 'retained_20']]

print("[Step 2] Loading global features...")
df_normdeg = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/normalized_degree_2020.csv")
df_weighted = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/weighted_degree_2020.csv")
df_clustering = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/global_user_clustering.csv")
df_betweenness = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/betweenness_centrality_2020.csv")

# Rename to 'address'
df_normdeg.rename(columns={"user": "address", "normalized_degree": "degree_global"}, inplace=True)
df_weighted.rename(columns={"user": "address", "weighted_degree": "weighted_degree_global"}, inplace=True)
df_clustering.rename(columns={"user": "address", "global_clustering": "clustering_global"}, inplace=True)
df_betweenness.rename(columns={"user": "address", "betweenness_centrality": "betweenness_global"}, inplace=True)

print("[Step 3] Loading community features...")
df_community = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/community_user_centralities.csv")
# Drop any label columns if they exist
df_community = df_community.loc[:, ~df_community.columns.str.startswith('retained_')]
df_community = df_community.loc[:, ~df_community.columns.str.startswith('not_retained_')]

print("[Step 4] Loading motif features...")
df_motif_details = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_details_2020.csv")
df_motif_timegaps = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_timegaps_2020.csv")

# Aggregate then pivot
df_motif_details = df_motif_details.groupby(['user', 'motif_type'])['user_count'].sum().unstack().reset_index()
df_motif_details.rename(columns={"user": "address"}, inplace=True)
df_motif_details.columns = ['address'] + [f"{col}_count" for col in df_motif_details.columns if col != 'address']

df_motif_timegaps = df_motif_timegaps.groupby(['user', 'motif_type'])['avg_time_gap_sec'].mean().unstack().reset_index()
df_motif_timegaps.rename(columns={"user": "address"}, inplace=True)
df_motif_timegaps.columns = ['address'] + [f"{col}_gapdays" for col in df_motif_timegaps.columns if col != 'address']
# Convert to days
for col in df_motif_timegaps.columns:
    if col.endswith('_gapdays'):
        df_motif_timegaps[col] = df_motif_timegaps[col] / (3600 * 24)

print("[Step 5] Loading cyclic pattern info...")
df_cyclic = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/cyclic_vs_retention_20.csv")
df_cyclic = df_cyclic[['address', 'cyclic_status']]
df_cyclic['cyclic_status'] = df_cyclic['cyclic_status'].map({'cyclic': 1, 'acyclic': 0})

print("[Step 6] Merging all features...")
df = df_retention.copy()
feature_sources = [
    df_normdeg, df_weighted, df_clustering, df_betweenness,
    df_community, df_motif_details, df_motif_timegaps, df_cyclic
]

for i, feature_df in enumerate(feature_sources):
    if 'address' not in feature_df.columns:
        raise ValueError(f"[ERROR] 'address' column missing in feature file #{i+1}")
    df = df.merge(feature_df, on='address', how='left')

# === Feature matrix and label ===
X = df.drop(columns=['address', 'retained_20'])
y = df['retained_20']

print("[Step 7] Splitting train/val/test...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("[Step 8] Applying SMOTE-ENN...")
smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)
print(f"After SMOTE-ENN: {X_train_res.shape}, Positives: {np.sum(y_train_res)}")

print("[Step 9] Running Optuna...")

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": 1.0,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42
    }
    model = XGBClassifier(**params)
    model.fit(X_train_res, y_train_res)
    val_proba = model.predict_proba(X_val)[:, 1]
    val_preds = (val_proba >= 0.5).astype(int)
    return f1_score(y_val, val_preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_params
print("[Best Params]:", best_params)

print("[Step 10] Training final model...")
model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train_res, y_train_res)

print("[Step 11] Threshold tuning...")
val_proba = model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"[Best Threshold]: {best_threshold:.2f}")

val_preds_opt = (val_proba >= best_threshold).astype(int)
test_proba = model.predict_proba(X_test)[:, 1]
test_preds_opt = (test_proba >= best_threshold).astype(int)

print("\n[Validation Results]")
print(classification_report(y_val, val_preds_opt))
print("ROC AUC (val):", roc_auc_score(y_val, val_proba))
print("Recall (val):", recall_score(y_val, val_preds_opt))

print("\n[Test Results]")
print(classification_report(y_test, test_preds_opt))
print("ROC AUC (test):", roc_auc_score(y_test, test_proba))
print("Recall (test):", recall_score(y_test, test_preds_opt))

# === Feature Importance ===
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="coolwarm")
plt.title("Feature Importance - Including Motifs and Cyclic Patterns")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "leiden_xgb_feature_importance_all.png"), dpi=300)
plt.show()

# === SHAP ===
explainer = shap.Explainer(model, X_train_res)
shap_values = explainer(X_val)

plt.figure()
shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "leiden_shap_summary_bar_all.png"), dpi=300)
plt.show()

plt.figure()
shap.summary_plot(shap_values, X_val, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "leiden_shap_summary_dot_all.png"), dpi=300)
plt.show()

print("✅ Model training complete using all features (retained_20 threshold).")
