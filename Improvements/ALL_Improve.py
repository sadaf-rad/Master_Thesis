# Your updated full script with Threshold Optimization (F1-focused)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

# Global centralities
df_degree = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/raw_degree_2020.csv", rename_col={'user': 'address'})
df_weighted = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/weighted_degree_2020.csv", rename_col={'user': 'address'})
df_clustering = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/clustering_coefficients_2020.csv")
df_betweenness = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/betweenness_centrality_2020.csv")
df_cyclic = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/cyclic_vs_retention_25.csv")[['address', 'cyclic_status']]
df_cyclic['cyclic_status'] = df_cyclic['cyclic_status'].map({'acyclic': 0, 'cyclic': 1})

# Community-level centralities
df_user_comm = load_and_clean("/home/s3986160/master-thesis/Improvements/community_user_centralities.csv", rename_col={'user': 'address'})
if 'retained_25' in df_user_comm.columns:
    df_user_comm.drop(columns=['retained_25'], inplace=True)
df_user_comm.rename(columns={
    "degree": "com_degree",
    "weighted_degree": "com_weighted_degree",
    "clustering": "com_clustering",
    "betweenness": "com_betweenness"
}, inplace=True)

# Community stats
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
motif_counts = pd.read_csv("/home/s3986160/master-thesis/Results/user_motif_details_2020.csv")
motif_counts.rename(columns={"user": "address"}, inplace=True)
motif_counts['address'] = motif_counts['address'].str.lower().str.strip()
motif_counts_pivot = motif_counts.pivot_table(index='address', columns='motif_type', values='user_count', aggfunc='sum')
motif_counts_pivot.columns = [f'motif_count_{col}' for col in motif_counts_pivot.columns]
motif_counts_pivot.reset_index(inplace=True)

motif_gaps = pd.read_csv("/home/s3986160/master-thesis/Results/user_motif_timegaps_2020.csv")
motif_gaps.rename(columns={"user": "address"}, inplace=True)
motif_gaps['address'] = motif_gaps['address'].str.lower().str.strip()
motif_gaps_pivot = motif_gaps.pivot_table(index='address', columns='motif_type', values='avg_time_gap_sec', aggfunc='mean')
motif_gaps_pivot.columns = [f'avg_gap_{col}' for col in motif_gaps_pivot.columns]
motif_gaps_pivot.reset_index(inplace=True)

# Demographics
df_demo = pd.read_csv("/home/s3986160/master-thesis/Results/global_users_demographics_2020_2021.csv")
df_demo['address'] = df_demo['address'].str.lower().str.strip()
df_demo = df_demo[['address', 'gender', 'area_name', 'business_type']]
df_demo[['gender', 'area_name', 'business_type']] = df_demo[['gender', 'area_name', 'business_type']].fillna('Unknown')

# === Step 2: Merge all ===
print("[2] Merging features...")
df = df_retention.copy()
df = df.merge(df_degree, on='address', how='left')
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

# Encode categoricals
print("[3] Encoding categorical variables...")
df = pd.get_dummies(df, columns=['gender', 'area_name', 'business_type'], drop_first=True)

# === Step 3: Split features and labels ===
print("[4] Preparing training set...")
X = df.drop(columns=['address', 'retained_25'])
y = df['retained_25']

print("[5] Splitting train/val/test...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# === Step 4: MICE imputation ===
print("[6] Applying MICE imputation...")
imputer = IterativeImputer(random_state=42)
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# === Step 5: SMOTE-ENN ===
print("[7] Balancing with SMOTE-ENN...")
smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train_imputed, y_train)

# === Step 6: Optuna tuning ===
print("[8] Running Optuna tuning...")
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
    preds = model.predict(X_val_imputed)
    return f1_score(y_val, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_params
print("[Best Params]:", best_params)

# === Step 7: Train final model ===
print("[9] Training final model...")
model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train_res, y_train_res)

# === Step 8: Evaluation with optimized threshold ===
print("[10] Evaluation...")
val_probs = model.predict_proba(X_val_imputed)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
f1s = 2 * (precision * recall) / (precision + recall + 1e-6)
best_thresh = thresholds[np.argmax(f1s)]
print(f"Best threshold: {best_thresh:.2f}")

val_preds = (val_probs >= best_thresh).astype(int)
print("\n📊 Validation Report")
print(classification_report(y_val, val_preds))
print("Val ROC AUC:", roc_auc_score(y_val, val_probs))

test_probs = model.predict_proba(X_test_imputed)[:, 1]
test_preds = (test_probs >= best_thresh).astype(int)
print("\n📊 Test Report")
print(classification_report(y_test, test_preds))
print("Test ROC AUC:", roc_auc_score(y_test, test_probs))

# === Step 9: Feature importance ===
print("[11] Feature importance plot...")
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values[:25], y=feat_imp.index[:25], palette="viridis")
plt.title("Top 25 Feature Importances - All Features")
plt.tight_layout()
plt.savefig("improve_leiden_xgb_feature_importance_model_leiden_mice.png", dpi=300)
plt.show()

print("\n✅ Model training complete with optimized threshold based on F1 score.")
