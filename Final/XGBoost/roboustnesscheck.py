# ================== Robustness to Retention Thresholds (Motifs + Cyclic) ==================
# Runs the same pipeline across all available retained_* columns and visualizes robustness.

# ------------------ Imports & Setup ------------------
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

# ------------------ Paths (edit if needed) ------------------
OUTPUT_DIR = "/home/s3986160/master-thesis/Retention/new definition/filtered/robustness_thresholds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base data
SARAFU_CSV   = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
RETENTION_CSV= "/home/s3986160/master-thesis/Retention/new definition/filtered/retention_AND.csv"

# Motif features (user-level)
MOTIF_COUNT_CSV   = "/home/s3986160/master-thesis/Results/user_motif_details_2020.csv"      # columns: user, motif_type, peers, user_count, global_count
MOTIF_TIMEGAPS_CSV= "/home/s3986160/master-thesis/Results/user_motif_timegaps_2020.csv"     # columns: user, motif_type, peers, avg_time_gap_sec

# ------------------ Optional: paste your tuned params here ------------------
BEST_PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    min_child_weight=1
)

RANDOM_STATE = 42
TEST_SIZE = 0.30  # 70/30 split; inside we split 50/50 for val/test on the held-out

# ------------------ Helper: build cyclic feature from SCC ------------------
def build_cyclic_feature(sarafu_csv):
    df = pd.read_csv(sarafu_csv, on_bad_lines='skip', encoding='utf-8')
    df['transfer_subtype'] = df['transfer_subtype'].astype(str).str.upper().str.strip()
    df['source'] = df['source'].astype(str).str.strip()
    df['target'] = df['target'].astype(str).str.strip()

    # keep only STANDARD and 2020; system-account filtering assumed already in your filtered runs,
    # but here we just keep STANDARD and year 2020 as in your scripts
    df = df[df['transfer_subtype'] == 'STANDARD'].copy()
    df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df_2020 = df[df['timestamp'].dt.year == 2020].copy()

    # directed graph
    G = nx.from_pandas_edgelist(df_2020, 'source', 'target', create_using=nx.DiGraph())
    # strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    # mark users in SCCs of size >= 2 as cyclic
    user2cyclic = {}
    for comp in sccs:
        flag = 1 if len(comp) >= 2 else 0
        for u in comp:
            user2cyclic[u] = flag

    # any isolated/singletons not in user2cyclic -> 0
    for u in G.nodes():
        if u not in user2cyclic:
            user2cyclic[u] = 0

    df_cyc = pd.DataFrame({'user': list(user2cyclic.keys()), 'is_cyclic': list(user2cyclic.values())})
    return df_cyc

# ------------------ Helper: build motif features ------------------
def build_motif_features(motif_count_csv, motif_timegaps_csv):
    # Counts
    dfc = pd.read_csv(motif_count_csv)
    # Keep only needed columns & pivot to user-level wide table
    # Create per-motif counts and totals
    # user_count: how many times user participates in motif_type
    dfc_user = dfc.groupby(['user', 'motif_type'])['user_count'].sum().reset_index()
    dfc_pivot = dfc_user.pivot(index='user', columns='motif_type', values='user_count').fillna(0)
    dfc_pivot.columns = [f"motif_count_{c}" for c in dfc_pivot.columns]
    # Also an overall total count across all motifs
    dfc_total = dfc_user.groupby('user')['user_count'].sum().rename('motif_count_total')

    # Time gaps
    dft = pd.read_csv(motif_timegaps_csv)
    # Aggregate avg_time_gap_sec by user and motif_type (mean)
    dft_agg = dft.groupby(['user', 'motif_type'])['avg_time_gap_sec'].mean().reset_index()
    dft_pivot = dft_agg.pivot(index='user', columns='motif_type', values='avg_time_gap_sec')
    # Missing gaps -> could be "no motif" so set a large/neutral or NaN. Use NaN; imputer will handle it.
    dft_pivot.columns = [f"motif_gap_{c}" for c in dft_pivot.columns]

    # Merge
    dfm = dfc_pivot.merge(dfc_total, left_index=True, right_index=True, how='outer')
    dfm = dfm.merge(dft_pivot, left_index=True, right_index=True, how='outer').reset_index().rename(columns={'index': 'user'})
    return dfm

# ------------------ Helper: collect labels and choose available thresholds ------------------
def load_labels(retention_csv):
    dfr = pd.read_csv(retention_csv, low_memory=False)
    # normalize id column
    if 'address' in dfr.columns and 'user' not in dfr.columns:
        dfr = dfr.rename(columns={'address': 'user'})
    # find all retained_* columns
    label_cols = [c for c in dfr.columns if c.startswith('retained_')]
    # keep only binary 0/1 columns
    valids = []
    for c in label_cols:
        vals = dfr[c].dropna().unique()
        if set(vals).issubset({0, 1}) and len(vals) > 0:
            valids.append(c)
    return dfr[['user'] + valids].copy(), sorted(valids, key=lambda x: int(x.split('_')[-1]))

# ------------------ Helper: train & evaluate one threshold ------------------
def run_one_threshold(dfX_base, dflab, label_col, random_state=RANDOM_STATE):
    df = dfX_base.merge(dflab[['user', label_col]], on='user', how='inner').copy()
    df = df.dropna(subset=[label_col])
    y = df[label_col].astype(int).values
    X = df.drop(columns=['user', label_col])

    # Standardize *counts* scale effects a bit (motifs counts can be heavy-tailed); imputer follows
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )
    # Split held-out temp into val/test (50/50)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # Impute (fit on train only)
    imputer = IterativeImputer(random_state=random_state, sample_posterior=False, max_iter=10, initial_strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_val   = imputer.transform(X_val)
    X_test  = imputer.transform(X_test)

    # Handle imbalance on train only
    # Compute scale_pos_weight from train distribution (for XGB) AND apply SMOTE-ENN for the tree learner
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = max(1.0, neg / max(1, pos))

    sampler = SMOTEENN(random_state=random_state)
    X_train_bal, y_train_bal = sampler.fit_resample(X_train, y_train)

    # XGB
    xgb = XGBClassifier(
        **BEST_PARAMS,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=spw,
        tree_method='hist'
    )

    # Calibrate with Platt scaling on validation set
    # Fit base model on the balanced train, then calibrate using validation
    xgb.fit(X_train_bal, y_train_bal)
    calibrator = CalibratedClassifierCV(xgb, method='sigmoid', cv='prefit')
    calibrator.fit(X_val, y_val)

    # Evaluate on test (probabilities)
    p_test = calibrator.predict_proba(X_test)[:, 1]

    # report metrics at default 0.5 AND the F1-optimal threshold (picked on validation)
    p_val = calibrator.predict_proba(X_val)[:, 1]

    # F1-optimal threshold from validation
    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = []
    for t in thresholds:
        yv_pred = (p_val >= t).astype(int)
        f1s.append(f1_score(y_val, yv_pred, zero_division=0))
    best_idx = int(np.argmax(f1s))
    t_star = float(thresholds[best_idx])

    # Test metrics @ 0.5
    y_pred_05 = (p_test >= 0.5).astype(int)
    metrics_05 = dict(
        roc_auc = roc_auc_score(y_test, p_test),
        pr_auc  = average_precision_score(y_test, p_test),
        f1      = f1_score(y_test, y_pred_05, zero_division=0),
        recall  = recall_score(y_test, y_pred_05, zero_division=0),
        precision = precision_score(y_test, y_pred_05, zero_division=0),
        thr_used = 0.5
    )

    # Test metrics @ t_star
    y_pred_star = (p_test >= t_star).astype(int)
    metrics_star = dict(
        roc_auc = roc_auc_score(y_test, p_test),  # same proba, so same AUC
        pr_auc  = average_precision_score(y_test, p_test),
        f1      = f1_score(y_test, y_pred_star, zero_division=0),
        recall  = recall_score(y_test, y_pred_star, zero_division=0),
        precision = precision_score(y_test, y_pred_star, zero_division=0),
        thr_used = t_star
    )

    # also keep class prevalence for this label definition
    prevalence = df[label_col].mean()

    return {
        'label': label_col,
        'prevalence': prevalence,
        'thr_f1_opt': t_star,
        'test_roc_auc@t*': metrics_star['roc_auc'],
        'test_pr_auc@t*': metrics_star['pr_auc'],
        'test_f1@t*': metrics_star['f1'],
        'test_recall@t*': metrics_star['recall'],
        'test_precision@t*': metrics_star['precision'],
        'test_roc_auc@0.5': metrics_05['roc_auc'],
        'test_pr_auc@0.5': metrics_05['pr_auc'],
        'test_f1@0.5': metrics_05['f1'],
        'test_recall@0.5': metrics_05['recall'],
        'test_precision@0.5': metrics_05['precision'],
    }

# ------------------ Main ------------------
if __name__ == "__main__":
    print("[1] Building features...")
    df_cyc = build_cyclic_feature(SARAFU_CSV)        # user, is_cyclic
    df_motifs = build_motif_features(MOTIF_COUNT_CSV, MOTIF_TIMEGAPS_CSV)  # user + motif_* columns

    # Merge to one feature set
    dfX = df_motifs.merge(df_cyc, on='user', how='outer')
    # Optional: fill motif counts with 0 (no motif observed), leave gaps as NaN
    count_cols = [c for c in dfX.columns if c.startswith('motif_count_')]
    dfX[count_cols] = dfX[count_cols].fillna(0)

    # Load all labels and discover available retained_* columns
    print("[2] Loading labels and finding thresholds...")
    dflab, label_cols = load_labels(RETENTION_CSV)
    if not label_cols:
        raise RuntimeError("No retained_* columns found in retention_AND.csv")

    print(f"    Found thresholds: {label_cols}")

    # Restrict to thresholds you care about (optional):
    # keep = {'retained_10','retained_15','retained_20','retained_25','retained_30'}
    # label_cols = [c for c in label_cols if c in keep and c in dflab.columns]

    # Run all thresholds
    print("[3] Training and evaluating across thresholds...")
    rows = []
    for lab in label_cols:
        print(f"    -> {lab}")
        res = run_one_threshold(dfX, dflab, lab)
        rows.append(res)

    # Save results
    df_results = pd.DataFrame(rows)
    # Sort by numeric threshold
    df_results['thr_pct'] = df_results['label'].str.split('_').str[-1].astype(int)
    df_results = df_results.sort_values('thr_pct')
    res_path = os.path.join(OUTPUT_DIR, "robustness_threshold_results.csv")
    df_results.to_csv(res_path, index=False)
    print("Saved:", res_path)

    # ------------------ Plots ------------------
    sns.set_context("talk")
    # 1) Performance vs threshold (F1/Recall/Precision/ROC AUC at t* )
    plt.figure(figsize=(9,6))
    plt.plot(df_results['thr_pct'], df_results['test_f1@t*'], marker='o', label='F1 ')
    plt.plot(df_results['thr_pct'], df_results['test_recall@t*'], marker='o', label='Recall ')
    plt.plot(df_results['thr_pct'], df_results['test_precision@t*'], marker='o', label='Precision ')
    plt.plot(df_results['thr_pct'], df_results['test_roc_auc@t*'], marker='o', label='ROC AUC')
    plt.xlabel("Retention threshold (%)")
    plt.ylabel("Score")
    plt.title("Robustness across retention thresholds")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    out_plot1 = os.path.join(OUTPUT_DIR, "robustness_metrics_vs_threshold.png")
    plt.tight_layout(); plt.savefig(out_plot1, dpi=300); plt.close()

    # 2) Positive class prevalence vs threshold
    plt.figure(figsize=(8,5))
    plt.plot(df_results['thr_pct'], df_results['prevalence'], marker='s')
    plt.xlabel("Retention threshold (%)")
    plt.ylabel("Share of retained users")
    plt.title("Class prevalence vs retention threshold")
    plt.grid(True, linestyle=':', alpha=0.5)
    out_plot2 = os.path.join(OUTPUT_DIR, "prevalence_vs_threshold.png")
    plt.tight_layout(); plt.savefig(out_plot2, dpi=300); plt.close()

    # 3) Compare F1 at 0.5 vs. F1 at t* (optional)
    plt.figure(figsize=(8,5))
    plt.plot(df_results['thr_pct'], df_results['test_f1@0.5'], marker='o', label='F1 @0.5')
    plt.plot(df_results['thr_pct'], df_results['test_f1@t*'], marker='o', label='F1 @t* (val-optimal)')
    plt.xlabel("Retention threshold (%)")
    plt.ylabel("F1-score")
    plt.title("Decision thresholding effect across retention definitions")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    out_plot3 = os.path.join(OUTPUT_DIR, "f1_fixed_vs_optimal.png")
    plt.tight_layout(); plt.savefig(out_plot3, dpi=300); plt.close()

    print("✅ Done.")
    print("  - Results CSV:", res_path)
    print("  - Plots:")
    print("    •", out_plot1)
    print("    •", out_plot2)
    print("    •", out_plot3)
