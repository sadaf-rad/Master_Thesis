import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_recall_curve, f1_score
from sklearn.inspection import permutation_importance
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

# === Output directory ===
output_dir = "./output/plots_threshold_analysis"
os.makedirs(output_dir, exist_ok=True)

print("[Step 1] Loading all features...")

# === Load files ===
df_retention = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/retention_AND.csv")
df_normdeg = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/normalized_degree_2020.csv")
df_weighted = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/weighted_degree_2020.csv")
df_clustering = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/global_user_clustering.csv")
df_betweenness = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/betweenness_centrality_2020.csv")
df_user_comm = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/community_user_centralities.csv")
df_motif = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_details_2020.csv")
df_gaps = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_timegaps_2020.csv")
df_cyclic = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/cyclic_users_2020.csv")
df_demo = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/standard_users_demographics.csv")

# === Clean addresses ===
for df_ in [df_retention, df_normdeg, df_weighted, df_clustering, df_betweenness, df_user_comm, df_motif, df_gaps, df_cyclic, df_demo]:
    addr_col = 'user' if 'user' in df_.columns else 'address'
    df_[addr_col] = df_[addr_col].astype(str).str.lower().str.strip()

# === Base with label ===
df = df_retention[["address", "retained_20"]].copy()

# === Global Features ===
df_normdeg.rename(columns={"user": "address", "normalized_degree": "degree_global"}, inplace=True)
df_weighted.rename(columns={"user": "address", "weighted_degree": "weighted_degree_global"}, inplace=True)
df_clustering.rename(columns={"user": "address", "global_clustering": "clustering_global"}, inplace=True)
df_betweenness.rename(columns={"user": "address", "betweenness_centrality": "betweenness_global"}, inplace=True)

df = df.merge(df_normdeg[["address", "degree_global"]], on="address", how="left")
df = df.merge(df_weighted[["address", "weighted_degree_global"]], on="address", how="left")
df = df.merge(df_clustering[["address", "clustering_global"]], on="address", how="left")
df = df.merge(df_betweenness[["address", "betweenness_global"]], on="address", how="left")

# === Community Features ===
df_user_comm.rename(columns={
    "user": "address",
    "degree": "degree_community",
    "weighted_degree": "weighted_degree_community",
    "clustering": "clustering_community",
    "betweenness": "betweenness_community",
    "community_size": "community_size",
    "consensus_community_id": "community_id"
}, inplace=True)

df = df.merge(df_user_comm[["address", "degree_community", "weighted_degree_community",
                            "clustering_community", "betweenness_community", "community_size", "community_id"]],
              on="address", how="left")

# === Motif Features ===
motif_counts = df_motif.groupby(["user", "motif_type"])["user_count"].sum().unstack().fillna(0)
motif_counts.columns = [f"{col}_motif_count" for col in motif_counts.columns]
motif_counts.reset_index(inplace=True)
motif_counts.rename(columns={"user": "address"}, inplace=True)
df = df.merge(motif_counts, on="address", how="left")

motif_gaps = df_gaps.groupby(["user", "motif_type"])["avg_time_gap_sec"].mean().unstack().fillna(0)
motif_gaps.columns = [f"{col}_avg_gap_sec" for col in motif_gaps.columns]
motif_gaps.reset_index(inplace=True)
motif_gaps.rename(columns={"user": "address"}, inplace=True)
df = df.merge(motif_gaps, on="address", how="left")

# === Cyclic Features ===
df_cyclic.rename(columns={"status": "cyclic_status"}, inplace=True)
df = df.merge(df_cyclic[["address", "cyclic_status"]], on="address", how="left")
df["cyclic_status"] = LabelEncoder().fit_transform(df["cyclic_status"].fillna("unknown"))

# === Demographic Features ===
df = df.merge(df_demo, on="address", how="left")
demo_cols = ["gender", "area_name", "area_type", "business_type", "held_roles"]
for col in demo_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str).fillna("unknown"))

# === Prepare Data ===
X = df.drop(columns=["address", "retained_20", "community_id"])
y = df["retained_20"]

print("[Step 2] Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

print("[Step 3] Imputing missing values...")
imputer = IterativeImputer(random_state=42)
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

print("[Step 4] Applying SMOTE-ENN...")
smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train_imputed, y_train)

print("[Step 5] Running Optuna...")
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
    preds = model.predict_proba(X_val_imputed)[:, 1]
    return f1_score(y_val, (preds >= 0.5).astype(int))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_params
print("[Best Params]:", best_params)

print("[Step 6] Training final model...")
neg, pos = np.bincount(y_train_res)
scale_pos_weight = neg / pos
print(f"[Class Imbalance] Negative: {neg}, Positive: {pos}, scale_pos_weight: {scale_pos_weight:.2f}")

model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight)
model.fit(X_train_res, y_train_res)

print("[Step 7] Threshold tuning...")
val_proba = model.predict_proba(X_val_imputed)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"[Best Threshold]: {best_threshold:.2f}")

val_preds = (val_proba >= best_threshold).astype(int)
test_proba = model.predict_proba(X_test_imputed)[:, 1]
test_preds = (test_proba >= best_threshold).astype(int)

print("\n[Validation Results]")
print(classification_report(y_val, val_preds))
print("ROC AUC (val):", roc_auc_score(y_val, val_proba))
print("Recall (val):", recall_score(y_val, val_preds))

print("\n[Test Results]")
print(classification_report(y_test, test_preds))
print("ROC AUC (test):", roc_auc_score(y_test, test_proba))
print("Recall (test):", recall_score(y_test, test_preds))

pd.DataFrame({"true_label": y_test, "predicted_proba": test_proba, "predicted_label": test_preds}).to_csv(
    os.path.join(output_dir, "test_predictions.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.axvline(best_threshold, linestyle="--", color="gray", label=f"Threshold = {best_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold - All Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "allprecision_recall_threshold.png"), dpi=300)
plt.show()

# === Step 8: Permutation Importance ===
print("[Step 8] Running permutation importance...")
perm_result = permutation_importance(model, X_val_imputed, y_val, n_repeats=10, random_state=42, n_jobs=-1)

perm_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std
}).sort_values(by="importance_mean", ascending=False)

perm_df.to_csv(os.path.join(output_dir, "permutation_importance_all_features.csv"), index=False)

plt.figure(figsize=(10, 6))
plt.barh(perm_df["feature"][:15][::-1], perm_df["importance_mean"][:15][::-1])
plt.xlabel("Permutation Importance (Mean)")
plt.title("Top 15 Feature Importances (Permutation)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "permutation_importance_top15.png"), dpi=300)
plt.show()

print("✅ Model training and feature importance completed.")
