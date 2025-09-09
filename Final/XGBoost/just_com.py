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

print("[Step 1] Loading community features...")

df_retention = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/retention_AND.csv")
df_comm = pd.read_csv("/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/community_user_centralities.csv")

df_comm.rename(columns={
    "user": "address",
    "degree": "degree_community",
    "weighted_degree": "weighted_degree_community",
    "clustering": "clustering_community",
    "betweenness": "betweenness_community",
    "community_size": "community_size"
}, inplace=True)

df = df_retention[["address", "retained_20"]].copy()
df = df.merge(
    df_comm[["address", "degree_community", "weighted_degree_community",
             "clustering_community", "betweenness_community", "community_size"]],
    on="address", how="left"
)

X = df.drop(columns=["address", "retained_20"])
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

# === Compute scale_pos_weight from resampled class balance ===
pos = int(np.sum(y_train_res == 1))
neg = int(np.sum(y_train_res == 0))
scale_pos_weight = neg / pos if pos > 0 else 1.0

print("[Step 5] Running Optuna...")
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": scale_pos_weight,
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
best_params["scale_pos_weight"] = scale_pos_weight
print("[Best Params]:", best_params)

print("[Step 6] Training final model and calibrating...")
# Fit base model first so we can use feature_importances_ and SHAP
base_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
base_model.fit(X_train_res, y_train_res)

# Calibrate on validation set (prefit)
calibrated_model = CalibratedClassifierCV(base_estimator=base_model, method="sigmoid", cv='prefit')
calibrated_model.fit(X_val_imputed, y_val)

print("[Step 7] Threshold tuning...")
val_proba = calibrated_model.predict_proba(X_val_imputed)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"[Best Threshold]: {best_threshold:.2f}")

val_preds = (val_proba >= best_threshold).astype(int)
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

# === Save test predictions ===
df_test_results = pd.DataFrame({
    "address": df.loc[X_test.index, "address"].values,
    "true_label": y_test.values,
    "predicted_proba": test_proba,
    "predicted_label": test_preds
})
df_test_results.to_csv(os.path.join(output_dir, "test_predictions_community_model.csv"), index=False)

# -------------------------------------------------------
# === Plot naming to match Table 3 (display only) ===
DISPLAY_LABELS = {
    "degree_community": "degree",
    "weighted_degree_community": "weighted degree",
    "clustering_community": "clustering coefficient",
    "betweenness_community": "betweenness",
    "community_size": "community size"
}
# -------------------------------------------------------

# === PR curve plot ===
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.axvline(best_threshold, linestyle="--", color="gray", label=f"Threshold = {best_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold - Community Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cprecision_recall_threshold.png"), dpi=300)
plt.show()

# === Feature importance (use fitted base_model) ===
importances = base_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="coolwarm")
ax.set_yticklabels([DISPLAY_LABELS.get(f, f) for f in feat_imp.index])
ax.set_title("Feature Importance - Community Only")
ax.set_xlabel("Importance Score")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "xgb_feature_importance_community.png"), dpi=300)
plt.show()

# === SHAP using the fitted base_model ===
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_val_imputed)

# Use renamed copy for SHAP labels
X_val_plot = pd.DataFrame(X_val_imputed, columns=[DISPLAY_LABELS.get(c, c) for c in X.columns])

plt.figure()
shap.summary_plot(shap_values, X_val_plot, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_bar_community.png"), dpi=300)
plt.show()

plt.figure()
shap.summary_plot(shap_values, X_val_plot, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_dot_community.png"), dpi=300)
plt.show()

print("✅ Community-only model complete: calibrated predictions + Table-3-consistent labels + fixed feature importance generation.")
