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

# === Only demographics ===
df_demo = pd.read_csv("/home/s3986160/master-thesis/Results/global_users_demographics_2020_2021.csv")
df_demo['address'] = df_demo['address'].str.lower().str.strip()
df_demo = df_demo[['address', 'gender', 'area_name', 'business_type']]
df_demo[['gender', 'area_name', 'business_type']] = df_demo[['gender', 'area_name', 'business_type']].fillna('Unknown')

# === Step 2: Merge ===
print("[2] Merging features...")
df = df_retention.merge(df_demo, on='address', how='left')

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
plt.title("Top 25 Feature Importances - Demographics Only")
plt.tight_layout()
plt.savefig("demographics_only_feature_importance.png", dpi=300)
plt.show()

print("\n✅ Model training complete with demographics only.")
