import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_recall_curve, f1_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

# === Output directory ===
output_dir = "./output/plots_threshold_analysis"
os.makedirs(output_dir, exist_ok=True)

print("[Step 1] Loading all features...")

# === Load all files ===
df_retention = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/retention_AND.csv")
df_normdeg = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/normalized_degree_2020.csv")
df_weighted = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/weighted_degree_2020.csv")
df_clustering = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/global_user_clustering.csv")
df_betweenness = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/betweenness_centrality_2020.csv")
df_user_comm = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/community_user_centralities.csv")
df_motif = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_details_2020.csv")
df_gaps = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_timegaps_2020.csv")
df_cyclic = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/cyclic_users_2020.csv")

# === Merge base with label ===
df = df_retention[['address', 'retained_20']].copy()

# === Global features ===
df_normdeg.rename(columns={"user": "address", "normalized_degree": "degree_global"}, inplace=True)
df_weighted.rename(columns={"user": "address", "weighted_degree": "weighted_degree_global"}, inplace=True)
df_clustering.rename(columns={"user": "address", "global_clustering": "clustering_global"}, inplace=True)
df_betweenness.rename(columns={"user": "address", "betweenness_centrality": "betweenness_global"}, inplace=True)

df = df.merge(df_normdeg[["address", "degree_global"]], on="address", how="left")
df = df.merge(df_weighted[["address", "weighted_degree_global"]], on="address", how="left")
df = df.merge(df_clustering[["address", "clustering_global"]], on="address", how="left")
df = df.merge(df_betweenness[["address", "betweenness_global"]], on="address", how="left")

# === Community features ===
df_user_comm.rename(columns={
    "user": "address",
    "degree": "degree_community",
    "weighted_degree": "weighted_degree_community",
    "clustering": "clustering_community",
    "betweenness": "betweenness_community",
    "community_size": "community_size",
    "consensus_community_id": "community_id"
}, inplace=True)

df = df.merge(
    df_user_comm[["address", "degree_community", "weighted_degree_community",
                  "clustering_community", "betweenness_community", "community_size", "community_id"]],
    on="address", how="left"
)

# === Motif counts ===
motif_counts = df_motif.groupby(["user", "motif_type"])["user_count"].sum().unstack().fillna(0)
motif_counts.columns = [f"{col}_motif_count" for col in motif_counts.columns]
motif_counts.reset_index(inplace=True)
motif_counts.rename(columns={"user": "address"}, inplace=True)
df = df.merge(motif_counts, on="address", how="left")

# === Motif completion times ===
motif_gaps = df_gaps.groupby(["user", "motif_type"])["avg_time_gap_sec"].mean().unstack().fillna(0)
motif_gaps.columns = [f"{col}_avg_gap_sec" for col in motif_gaps.columns]
motif_gaps.reset_index(inplace=True)
motif_gaps.rename(columns={"user": "address"}, inplace=True)
df = df.merge(motif_gaps, on="address", how="left")

# === Cyclic status ===
df = df.merge(df_cyclic, on="address", how="left")
df["cyclic_status"] = LabelEncoder().fit_transform(df["cyclic_status"].fillna("unknown"))

# === Final dataset ===
X = df.drop(columns=["address", "retained_20", "community_id"])
y = df["retained_20"]

print("[Step 2] Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# === Imputation ===
print("[Step 3] Imputing missing values with IterativeImputer...")
imputer = IterativeImputer(random_state=42)
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# === Resampling ===
print("[Step 4] Applying SMOTE-ENN...")
smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train_imputed, y_train)

# === Optuna tuning ===
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

print("[Step 5] Running Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_params
print("[Best Params]:", best_params)

# === Final model training + calibration ===
print("[Step 6] Training final model...")
neg, pos = np.bincount(y_train_res)
scale_pos_weight = neg / pos if pos > 0 else 1.0
print(f"[Class Imbalance] Negative: {neg}, Positive: {pos}, scale_pos_weight: {scale_pos_weight:.2f}")

base_model = XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_pos_weight
)
base_model.fit(X_train_res, y_train_res)

calibrated_model = CalibratedClassifierCV(base_estimator=base_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_val_imputed, y_val)

# === Threshold tuning ===
print("[Step 7] Threshold tuning...")
val_proba = calibrated_model.predict_proba(X_val_imputed)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"[Best Threshold]: {best_threshold:.2f}")

val_preds = (val_proba >= best_threshold).astype(int)
test_proba = calibrated_model.predict_proba(X_test_imputed)[:, 1]
test_preds = (test_proba >= best_threshold).astype(int)

# === Evaluation ===
print("\n[Validation Results]")
print(classification_report(y_val, val_preds))
print("ROC AUC (val):", roc_auc_score(y_val, val_proba))
print("Recall (val):", recall_score(y_val, val_preds))

print("\n[Test Results]")
print(classification_report(y_test, test_preds))
print("ROC AUC (test):", roc_auc_score(y_test, test_proba))
print("Recall (test):", recall_score(y_test, test_preds))

# === Save predictions ===
pd.DataFrame({
    "true_label": y_test,
    "predicted_proba": test_proba,
    "predicted_label": test_preds
}).to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

# -------------------------------------------------------
# === Display name mapping for plots (Table 3 style, EXACT motif names) ===
DISPLAY_LABELS = {
    # global
    "degree_global": "degree (global)",
    "weighted_degree_global": "weighted degree (global)",
    "clustering_global": "clustering coefficient (global)",
    "betweenness_global": "betweenness (global)",
    # community
    "degree_community": "degree (community)",
    "weighted_degree_community": "weighted degree (community)",
    "clustering_community": "clustering coefficient (community)",
    "betweenness_community": "betweenness (community)",
    "community_size": "community size (community)",
    # cyclic
    "cyclic_status": "cyclic status",
    # motifs — EXACT names requested (no 'avg sec')
    "cycle_motif_count": "3-Node Motif Count",
    "cycle_avg_gap_sec": "3-Node Motif Completion Time",
    "reciprocal_motif_count": "Reciprocal Motif Count",
    "reciprocal_avg_gap_sec": "Reciprocal Completion Time"
}
# -------------------------------------------------------

def rename_for_plot(columns):
    return [DISPLAY_LABELS.get(c, c.replace("_", " ")) for c in columns]

# === PR curve plot ===
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.axvline(best_threshold, linestyle="--", color="gray", label=f"Threshold = {best_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold - With Motifs and Cyclic Pattern")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "precision_recall_threshold.png"), dpi=300)
plt.show()

# === Feature importance ===
importances = base_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(25)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="coolwarm")
ax.set_yticklabels(rename_for_plot(feat_imp.index))
ax.set_title("Feature Importances - With Motifs and Cyclic Pattern")
ax.set_xlabel("Importance Score")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "3feature_importance_top25.png"), dpi=300)
plt.show()

# === SHAP analysis ===
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_val_imputed)

X_val_plot = pd.DataFrame(X_val_imputed, columns=rename_for_plot(X.columns))

plt.figure()
shap.summary_plot(shap_values, X_val_plot, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=300)
plt.show()

plt.figure()
shap.summary_plot(shap_values, X_val_plot, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_dot.png"), dpi=300)
plt.show()

print("✅ All-features model complete: fitted safely + motif labels exactly as requested.")
# === [Step 2.6] Summary: users, features, edges, nodes per set ===
df_txn = pd.read_csv("/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv",
                     on_bad_lines='skip', encoding='utf-8')

# Clean transaction data
df_txn['transfer_subtype'] = df_txn['transfer_subtype'].astype(str).str.upper().str.strip()
df_txn['source'] = df_txn['source'].astype(str).str.strip()
df_txn['target'] = df_txn['target'].astype(str).str.strip()
df_txn['timestamp'] = pd.to_datetime(df_txn['timeset'], errors='coerce')
df_txn = df_txn.dropna(subset=['timestamp'])
df_txn = df_txn[df_txn['transfer_subtype'] == 'STANDARD']
df_txn = df_txn[df_txn['timestamp'].dt.year == 2020].copy()

# Extract addresses for splits
X_all_with_address = df[["address"]].reset_index()
X_train_addr = X_all_with_address.loc[X_train.index, "address"].values
X_val_addr   = X_all_with_address.loc[X_val.index, "address"].values
X_test_addr  = X_all_with_address.loc[X_test.index, "address"].values

# Helper function to count edges and nodes
def txn_stats(name, addr_list):
    txn_subset = df_txn[df_txn['source'].isin(addr_list) | df_txn['target'].isin(addr_list)]
    nodes = set(txn_subset['source']).union(set(txn_subset['target']))
    print(f"[{name}] Users: {len(addr_list):,} | Edges: {len(txn_subset):,} | Nodes: {len(nodes):,}")

print("[Step 2.6] Split Summary")
print(f"Number of features: {X.shape[1]}")
txn_stats("Train Set", X_train_addr)
txn_stats("Validation Set", X_val_addr)
txn_stats("Test Set", X_test_addr)
