import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_recall_curve, f1_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

# === Output directory ===
output_dir = "/home/s3986160/master-thesis/Retention/new definition/filtered/models/modified_models"
os.makedirs(output_dir, exist_ok=True)

print("[Step 1] Loading global features...")

# === Load labeled global features and retention ===
df_retention = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/retention_AND.csv")
df_normdeg = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/normalized_degree_2020.csv")
df_weighted = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/weighted_degree_2020.csv")
df_clustering = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/global_user_clustering.csv")
df_betweenness = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/betweenness_centrality_2020.csv")

# === Rename columns to unified feature names (internal names; not shown in plots) ===
df_normdeg.rename(columns={"user": "address", "normalized_degree": "degree_global"}, inplace=True)
df_weighted.rename(columns={"user": "address", "weighted_degree": "weighted_degree_global"}, inplace=True)
df_clustering.rename(columns={"user": "address", "global_clustering": "clustering_global"}, inplace=True)
df_betweenness.rename(columns={"user": "address", "betweenness_centrality": "betweenness_global"}, inplace=True)

# === Merge into one dataframe ===
df = df_retention[['address', 'retained_20']].copy()
df = df.merge(df_normdeg[['address', 'degree_global']], on='address', how='left')
df = df.merge(df_weighted[['address', 'weighted_degree_global']], on='address', how='left')
df = df.merge(df_clustering[['address', 'clustering_global']], on='address', how='left')
df = df.merge(df_betweenness[['address', 'betweenness_global']], on='address', how='left')

# === Feature matrix and label ===
X = df.drop(columns=['address', 'retained_20'])
y = df['retained_20']

# === Train/val/test split ===
print("[Step 2] Splitting train/val/test...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# === SMOTE-ENN oversampling ===
print("[Step 3] Applying SMOTE-ENN...")
smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)
print(f"After SMOTE-ENN: {X_train_res.shape}, Positives: {np.sum(y_train_res)}")

# === Compute scale_pos_weight from original class ratio ===
class_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

# === Optuna search ===
print("[Step 4] Running Optuna...")

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": class_ratio,
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
best_params["scale_pos_weight"] = class_ratio
print("[Best Params]:", best_params)

# === Final model training ===
print("[Step 5] Training final model...")
model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train_res, y_train_res)

# === Calibrate model ===
print("[Step 6] Calibrating model...")
calibrated_model = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_val, y_val)

# === Threshold tuning ===
print("[Step 7] Threshold tuning...")
val_proba = calibrated_model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"[Best Threshold]: {best_threshold:.2f}")

# === Evaluation ===
val_preds_opt = (val_proba >= best_threshold).astype(int)
test_proba = calibrated_model.predict_proba(X_test)[:, 1]
test_preds_opt = (test_proba >= best_threshold).astype(int)

print("\n[Validation Results]")
print(classification_report(y_val, val_preds_opt))
print("ROC AUC (val):", roc_auc_score(y_val, val_proba))
print("Recall (val):", recall_score(y_val, val_preds_opt))

print("\n[Test Results]")
print(classification_report(y_test, test_preds_opt))
print("ROC AUC (test):", roc_auc_score(y_test, test_proba))
print("Recall (test):", recall_score(y_test, test_preds_opt))

# === Save test predictions ===
results_df = pd.DataFrame({
    "address": df.loc[X_test.index, "address"],
    "true_label": y_test.values,
    "predicted_proba": test_proba,
    "predicted_label": test_preds_opt
})
results_df.to_csv(os.path.join(output_dir, "test_predictions_global_model.csv"), index=False)

# -------------------------------------------------------
# === Plot naming to match Table 3 (display only) ===
# Map internal feature names -> exact table display names
DISPLAY_LABELS = {
    "degree_global": "degree",
    "weighted_degree_global": "weighted degree",
    "clustering_global": "clustering coefficient",
    "betweenness_global": "betweenness",
}
# -------------------------------------------------------

# === Plot PR curve ===
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.axvline(x=best_threshold, linestyle="--", color="gray", label=f"Threshold = {best_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold - Global Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "leiden_pr_threshold_global.png"), dpi=300)
plt.show()

# === Feature importance ===
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="coolwarm")
# Replace y tick labels with Table-3 names (order preserved)
ax.set_yticklabels([DISPLAY_LABELS.get(f, f) for f in feat_imp.index])
ax.set_title("Feature Importance - Global Only")
ax.set_xlabel("Importance Score")
ax.set_ylabel("")  # clean axis label
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "leiden_xgb_feature_importance_global.png"), dpi=300)
plt.show()

# === SHAP TreeExplainer ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Use a renamed copy for nicer SHAP labels; model inputs stay untouched
X_val_plot = X_val.copy()
X_val_plot.rename(columns=DISPLAY_LABELS, inplace=True)

plt.figure()
shap.summary_plot(shap_values, X_val_plot, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "leiden_shap_summary_bar_global.png"), dpi=300)
plt.show()

plt.figure()
shap.summary_plot(shap_values, X_val_plot, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "leiden_shap_summary_dot_global.png"), dpi=300)
plt.show()

print("✅ Model training complete with calibrated predictions, class-weight adjustment, and Table‑3‑consistent plot labels.")
