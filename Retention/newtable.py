import pandas as pd
import numpy as np
import os

# === Config (GLOBAL) ===
centrality_files = {
    "raw_degree": (
        "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/raw_degree_2020.csv",
        "raw_degree", "user"),
    "betweenness": (
        "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/betweenness_centrality_2020.csv",
        "betweenness_centrality", "user"),
    "clustering": (
        "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/global_user_clustering.csv",
        "global_clustering", "user"),
    "weighted_degree": (
        "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/weighted_degree_2020.csv",
        "weighted_degree", "user"),
}

RETENTION_PATH = "/home/s3986160/master-thesis/Retention/new definition/filtered/retention_AND.csv"
RETENTION_COL = "retained_20"
OUTPUT_DIR = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled"
MERGED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_retention_centrality_stats_20_log.csv")

# === Config (COMMUNITY) ===
COMMUNITY_FILE = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/community_user_centralities.csv"
COMMUNITY_LABEL_COL = "retained_20"
COMMUNITY_METRICS = ["degree", "weighted_degree", "clustering", "betweenness"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def mean_std_log(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce").clip(lower=0)
    log_vals = np.log1p(vals).dropna()
    return float(log_vals.mean()), float(log_vals.std(ddof=1)) if len(log_vals) > 1 else (np.nan, np.nan)

# === Load Retention ===
retention_df = pd.read_csv(RETENTION_PATH)[["address", RETENTION_COL]].rename(
    columns={"address": "user", RETENTION_COL: "retention_label"}
)

results = []

# =========================
# GLOBAL centrality stats
# =========================
for name, (path, centrality_col, id_col) in centrality_files.items():
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path).rename(columns={id_col: "user"})
    if centrality_col not in df.columns:
        continue
    merged = df.merge(retention_df, on="user", how="inner")

    for label in [0, 1]:
        sub = merged[merged["retention_label"] == label]
        m, s = mean_std_log(sub[centrality_col])
        results.append({
            "level": "global",
            "centrality": name,
            "metric_column": centrality_col,
            "retention_label": label,
            "count": len(sub),
            "mean_log": m,
            "std_log": s,
        })

# =========================
# COMMUNITY centrality stats
# =========================
if os.path.exists(COMMUNITY_FILE):
    cdf = pd.read_csv(COMMUNITY_FILE).rename(columns={COMMUNITY_LABEL_COL: "retention_label"})
    for metric in COMMUNITY_METRICS:
        for label in [0, 1]:
            sub = cdf[cdf["retention_label"] == label]
            m, s = mean_std_log(sub[metric])
            results.append({
                "level": "community",
                "centrality": metric,
                "metric_column": metric,
                "retention_label": label,
                "count": len(sub),
                "mean_log": m,
                "std_log": s,
            })

# =========================
# SAVE
# =========================
merged_df = pd.DataFrame(results)
merged_df.to_csv(MERGED_OUTPUT_FILE, index=False)
print(f"✅ Merged stats saved to: {MERGED_OUTPUT_FILE}")
