import pandas as pd
import numpy as np
import os

# === Paths ===
BASE = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled"
out_csv = f"{BASE}/centrality_stats_log.csv"

# === Load files ===
df_raw_deg = pd.read_csv(f"{BASE}/raw_degree_2020.csv")  # user, raw_degree
df_weighted = pd.read_csv(f"{BASE}/weighted_degree_2020.csv")  # user, weighted_degree, retained_20
df_betw = pd.read_csv(f"{BASE}/betweenness_centrality_2020.csv")  # user, betweenness_centrality, retained_20
df_deg_norm = pd.read_csv(f"{BASE}/degree_centrality_2020.csv")  # user, degree_centrality, retained_20
df_clust = pd.read_csv(f"{BASE}/global_user_clustering.csv")  # user, global_clustering, retained_20
df_comm = pd.read_csv(f"{BASE}/community_user_centralities.csv")  # includes degree, weighted_degree, clustering, betweenness, retained_20

# === Merge into one dataframe ===
dfs = [df_raw_deg, df_weighted, df_betw, df_deg_norm, df_clust, df_comm]
df = dfs[0]
for other in dfs[1:]:
    df = pd.merge(df, other, on="user", how="outer")

# === Keep only needed columns ===
keep_cols = [
    "user", "retained_20",
    "raw_degree", "weighted_degree", "betweenness_centrality",
    "degree_centrality", "global_clustering",
    "degree", "weighted_degree_y", "clustering", "betweenness"
]
df = df[[c for c in keep_cols if c in df.columns]].copy()

# === Rename for clarity ===
df = df.rename(columns={
    "weighted_degree": "weighted_degree_global",
    "weighted_degree_y": "weighted_degree_comm",
    "degree": "degree_comm",
    "clustering": "clustering_comm",
    "betweenness": "betweenness_comm"
})

# === Apply log transform only to skewed features ===
log_features = [
    "raw_degree", "weighted_degree_global", "weighted_degree_comm",
    "betweenness_centrality", "betweenness_comm"
]
for col in log_features:
    if col in df.columns:
        df[col] = np.log1p(df[col])  # log(1+x) transform

# === Compute descriptive stats in log scale ===
log_stats = df.groupby("retained_20").agg(['mean', 'std']).round(6)

# === Flatten MultiIndex ===
log_stats.columns = [f"{col[0]}_{col[1]}" for col in log_stats.columns]

# === Save result ===
log_stats.to_csv(out_csv)
print(f"✅ Log-scale centrality stats saved to {out_csv}")
print(log_stats.head())
