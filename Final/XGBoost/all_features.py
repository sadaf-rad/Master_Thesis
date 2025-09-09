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

# === Load files ===
df_retention = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/retention_AND.csv")
df_normdeg   = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/normalized_degree_2020.csv")
df_weighted  = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/weighted_degree_2020.csv")
df_clustering= pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/global_user_clustering.csv")
df_betweenness = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/betweenness_centrality_2020.csv")
df_user_comm = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/community_user_centralities.csv")
df_motif     = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_details_2020.csv")
df_gaps      = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_timegaps_2020.csv")
df_cyclic    = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/cyclic_users_2020.csv")
df_demo      = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/standard_users_demographics.csv")

# === Clean addresses ===
for df_ in [df_retention, df_normdeg, df_weighted, df_clustering, df_betweenness,
            df_user_comm, df_motif, df_gaps, df_cyclic, df_demo]:
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

df = df.merge(
    df_user_comm[["address", "degree_community", "weighted_degree_community",
                  "clustering_community", "betweenness_community", "community_size", "community_id"]],
    on="address", how="left"
)

# === Motif Features (rename to FINAL names at creation) ===
# Counts per motif
motif_counts = df_motif.groupby(["user", "motif_type"])["user_count"].sum().unstack().fillna(0)
counts_name_map = {
    "cycle_3node": "3-Node Motif Count",
    "cycle": "3-Node Motif Count",       # fallback if labeled 'cycle'
    "reciprocal": "Reciprocal Motif Count",
}
motif_counts.rename(columns={c: counts_name_map.get(c, c) for c in motif_counts.columns}, inplace=True)
motif_counts.reset_index(inplace=True)
motif_counts.rename(columns={"user": "address"}, inplace=True)
df = df.merge(motif_counts, on="address", how="left")

# Completion time per motif (seconds) — rename to final names (no 'avg sec')
motif_gaps = df_gaps.groupby(["user", "motif_type"])["avg_time_gap_sec"].mean().unstack().fillna(0)
gaps_name_map = {
    "cycle_3node": "3-Node Motif Completion Time",
    "cycle": "3-Node Motif Completion Time",
    "reciprocal": "Reciprocal Completion Time",
}
motif_gaps.rename(columns={c: gaps_name_map.get(c, c) for c in motif_gaps.columns}, inplace=True)
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
X_val_imputed   = imputer.transform(X_val)
X_test_imputed  = imputer.transform(X_test)

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
    mdl = XGBClassifier(**params)
    mdl.fit(X_train_res, y_train_res)
    preds = mdl.predict_proba(X_val_imputed)[:, 1]
    return f1_score(y_val, (preds >= 0.5).astype(int))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_params
print("[Best Params]:", best_params)

print("[Step 6] Training final model (fit first) and calibrating...")
neg = int(np.sum(y_train_res == 0))
pos = int(np.sum(y_train_res == 1))
scale_pos_weight = (neg / pos) if pos > 0 else 1.0
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

print("[Step 7] Threshold tuning...")
val_proba = calibrated_model.predict_proba(X_val_imputed)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"[Best Threshold]: {best_threshold:.2f}")

val_preds  = (val_proba >= best_threshold).astype(int)
test_proba = calibrated_model.predict_proba(X_test_imputed)[:, 1]
test_preds = (test_proba >= best_threshold).astype(int)

print("\n[Validation Results]")
print(classification_report(y_val, val_preds))
print("ROC AUC (val):", roc_auc_score(y_val, val_proba))
print("Recall (val):", recall_score(y_val, val_preds))

print("\n[Test Results]")
print(classification_report(y_test, test_preds))
print("ROC AUC (test):", roc_auc_score(y_test, test_proba))
print("Recall (test):", recall_score(y_test, test_preds))

pd.DataFrame(
    {"true_label": y_test, "predicted_proba": test_proba, "predicted_label": test_preds}
).to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

# -------------------------------------------------------
# Plotting
# Centralities & demographics get pretty labels; motif columns are already final names.
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
    "community_size": "community size ",
    # cyclic
    "cyclic_status": "cyclic status",
    # demographics
    "gender": "gender",
    "area_name": "area name",
    "area_type": "area type",
    "business_type": "business type",
    "held_roles": "held roles",
    # NOTE: motifs already have final names
}

def pretty(colnames):
    return [DISPLAY_LABELS.get(c, c) for c in colnames]

# === PR curve plot ===
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

# === Feature importance (use fitted base_model) ===
importances = base_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(25)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="coolwarm")
ax.set_yticklabels(pretty(feat_imp.index))
ax.set_title("Feature Importances - All Features")
ax.set_xlabel("Importance Score")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "8allfeature_importance_top25.png"), dpi=300)
plt.show()

# === SHAP analysis (use fitted base_model) ===
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_val_imputed)

X_val_plot = pd.DataFrame(X_val_imputed, columns=pretty(X.columns))

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

print("✅ Model training complete, with motif labels set at source to your exact Table-3 wording.")
