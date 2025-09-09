import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.combine import SMOTEENN
import xgboost as xgb
import optuna

print("[Step 1] Loading and cleaning files...")

def load_and_clean(path, rename_col=None):
    df = pd.read_csv(path)
    if rename_col:
        df.rename(columns=rename_col, inplace=True)
    df['address'] = df['address'].astype(str).str.lower().str.strip()
    return df

# --- Load datasets ---
df_retention = load_and_clean(
    "/home/s3986160/master-thesis/Retention/new definition/retention_AND.csv"
)[['address', 'retained_25']]
df_retention['retained_25'] = df_retention['retained_25'].astype(int)

df_com = load_and_clean(
    "/home/s3986160/master-thesis/Lastround /output/community_users_with_all_centralities.csv"
)[['address', 'community_id', 'community_size', 'degree', 'betweenness', 'clustering', 'weighted_degree']]

df_degree = load_and_clean(
    "/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/raw_degree_2020.csv",
    rename_col={'user': 'address'}
)

df_weighted = load_and_clean(
    "/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/weighted_degree_2020.csv",
    rename_col={'user': 'address'}
)

df_clustering = load_and_clean(
    "/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/clustering_coefficients_2020.csv"
)

df_betweenness = load_and_clean(
    "/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/betweenness_centrality_2020.csv"
)

df_cyclic = load_and_clean(
    "/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/cyclic_vs_retention_25.csv"
)[['address', 'cyclic_status']]
df_cyclic['cyclic_status'] = df_cyclic['cyclic_status'].map({'acyclic': 0, 'cyclic': 1})

motif_counts = pd.read_csv("/home/s3986160/master-thesis/Results/user_motif_details_2020.csv")
motif_counts.rename(columns={"user": "address"}, inplace=True)
motif_counts['address'] = motif_counts['address'].str.lower().str.strip()
motif_counts_pivot = motif_counts.pivot_table(index='address', columns='motif_type', values='user_count', aggfunc='sum').fillna(0)
motif_counts_pivot.columns = [f'motif_count_{col}' for col in motif_counts_pivot.columns]
motif_counts_pivot.reset_index(inplace=True)

motif_gaps = pd.read_csv("/home/s3986160/master-thesis/Results/user_motif_timegaps_2020.csv")
motif_gaps.rename(columns={"user": "address"}, inplace=True)
motif_gaps['address'] = motif_gaps['address'].str.lower().str.strip()
motif_gaps_pivot = motif_gaps.pivot_table(index='address', columns='motif_type', values='avg_time_gap_sec', aggfunc='mean').fillna(0)
motif_gaps_pivot.columns = [f'avg_gap_{col}' for col in motif_gaps_pivot.columns]
motif_gaps_pivot.reset_index(inplace=True)

df_demo = pd.read_csv("/home/s3986160/master-thesis/Results/global_users_demographics_2020_2021.csv")
df_demo['address'] = df_demo['address'].str.lower().str.strip()
df_demo = df_demo[['address', 'gender', 'area_name', 'business_type']]
df_demo[['gender', 'area_name', 'business_type']] = df_demo[['gender', 'area_name', 'business_type']].fillna('Unknown')

df_dates = pd.read_csv("/home/s3986160/master-thesis/Results/users_in_both_years_txn_dates.csv")
df_dates['address'] = df_dates['address'].str.lower().str.strip()
df_dates['first_txn'] = pd.to_datetime(df_dates['first_txn'], errors='coerce')
df_dates['last_txn'] = pd.to_datetime(df_dates['last_txn'], errors='coerce')
df_dates['first_txn_unix'] = df_dates['first_txn'].view('int64') // 10**9
df_dates['last_txn_unix'] = df_dates['last_txn'].view('int64') // 10**9

print("[Step 2] Merging all features...")
df = df_retention.merge(df_com, on='address', how='inner')
df = df.merge(df_degree, on='address', how='left')
df = df.merge(df_weighted, on='address', how='left')
df = df.merge(df_clustering, on='address', how='left')
df = df.merge(df_betweenness, on='address', how='left')
df = df.merge(df_cyclic, on='address', how='left')
df = df.merge(motif_counts_pivot, on='address', how='left')
df = df.merge(motif_gaps_pivot, on='address', how='left')
df = df.merge(df_demo, on='address', how='left')
df = df.merge(df_dates[['address', 'first_txn_unix', 'last_txn_unix']], on='address', how='left')

print("[Step 3] Handling missing values...")
df.fillna(0, inplace=True)
print(f"[DEBUG] Final merged dataset shape: {df.shape}")

print("[Step 4] Encoding categorical variables...")
df['community_id'] = df['community_id'].astype(str)
df = pd.get_dummies(df, columns=['community_id', 'gender', 'area_name', 'business_type'], drop_first=True)

X = df.drop(columns=['address', 'retained_25'])
y = df['retained_25'].astype(int)

print("[Step 5] Splitting into train/val/test...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("[Step 6] Applying SMOTE-ENN...")
smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)
print(f"After SMOTE-ENN: {X_train_res.shape}, Positives: {sum(y_train_res)}")

print("[Step 7] Tuning XGBoost with Optuna...")
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_res, y_train_res)
    preds = model.predict(X_val)
    return classification_report(y_val, preds, output_dict=True)['1']['f1-score']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
print("[Best Parameters]:", study.best_params)

print("[Step 8] Training final XGBoost model...")
best_model = xgb.XGBClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train_res, y_train_res)

print("[Step 9] Threshold optimization on validation set...")
val_proba = best_model.predict_proba(X_val)[:, 1]
best_thresh = 0.4
val_preds = (val_proba >= best_thresh).astype(int)
print("\nValidation Results")
print(classification_report(y_val, val_preds))
print("ROC AUC (val):", roc_auc_score(y_val, val_proba))

test_proba = best_model.predict_proba(X_test)[:, 1]
test_preds = (test_proba >= best_thresh).astype(int)
print("\nTest Results")
print(classification_report(y_test, test_preds))
print("ROC AUC (test):", roc_auc_score(y_test, test_proba))

print("[Step 10] Feature importance plot...")
importances = best_model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values[:25], y=feat_imp.index[:25], palette="viridis")
plt.title("Top 25 Feature Importances - All Features")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("shpxgfeature_importance_all_xgb.png", dpi=300)
plt.show()
print("✅ Feature importance plot saved.")

# Step 11: SHAP explanation
print("[Step 11] Calculating SHAP values for final model...")
explainer = shap.Explainer(best_model, X_train_res)
shap_values = explainer(X_val, check_additivity=False)

print("✅ Generating SHAP summary plot...")
shap.summary_plot(shap_values, X_val, max_display=25)

print("✅ Saving SHAP summary plot to file...")
plt.tight_layout()
plt.savefig("shap_summary_plot_xgb_all.png", dpi=300)
plt.close()

print("✅ SHAP explanation completed for final XGBoost model.")
