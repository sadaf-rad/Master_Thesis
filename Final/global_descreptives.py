import pandas as pd
import os

# === Config ===
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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "global_retention_centrality_stats_20.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Retention ===
retention_df = pd.read_csv(RETENTION_PATH)[["address", RETENTION_COL]]
retention_df = retention_df.rename(columns={"address": "user", RETENTION_COL: "retention_label"})

# === Process Each Centrality ===
results = []

for name, (path, centrality_col, id_col) in centrality_files.items():
    if not os.path.exists(path):
        print(f"[⚠️] Missing: {path}")
        continue

    df = pd.read_csv(path)
    if id_col not in df.columns or centrality_col not in df.columns:
        print(f"[⚠️] Missing columns in {name}: {id_col}, {centrality_col}")
        continue

    df = df.rename(columns={id_col: "user"})
    merged = df.merge(retention_df, on="user", how="inner")

    # Global stats
    global_stats = merged[centrality_col].describe().to_dict()
    global_stats.update({
        "centrality": name,
        "retention_label": "all",
        "count": len(merged)
    })
    results.append(global_stats)

    # Retained vs Not Retained
    for label in [0, 1]:
        subset = merged[merged["retention_label"] == label]
        stats = subset[centrality_col].describe().to_dict()
        stats.update({
            "centrality": name,
            "retention_label": label,
            "count": len(subset)
        })
        results.append(stats)

# === Save Output ===
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Centrality stats saved to: {OUTPUT_FILE}")
