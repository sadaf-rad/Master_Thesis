import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
global_path = "/home/s3986160/master-thesis/Lastround /output/global_retention_centrality_stats.csv"
community_path = "/home/s3986160/master-thesis/Lastround /output/overall_community_centrality_stats.csv"
output_dir = "/home/s3986160/master-thesis/Lastround /output/"
os.makedirs(output_dir, exist_ok=True)

# Load data
df_global = pd.read_csv(global_path)
df_community = pd.read_csv(community_path)

# Rename to match naming conventions
df_global["centrality"] = df_global["centrality"].replace({"raw_degree": "degree"})

# Filter global to 'all' only and keep required columns
df_global = df_global[df_global["retention_label"] == "all"]
df_global = df_global[["centrality", "mean", "std"]].copy()
df_global = df_global.rename(columns={"mean": "global_mean", "std": "global_std"})

# Prepare community data
df_community = df_community[["centrality", "mean", "std", "community_id"]].copy()
df_community = df_community.rename(columns={"mean": "community_mean", "std": "community_std"})

# Merge on centrality
merged = df_community.merge(df_global, on="centrality", how="left")

# Add comparison result
def compare_means(row, tol=1e-6):
    if abs(row["community_mean"] - row["global_mean"]) < tol:
        return "equal"
    elif row["community_mean"] > row["global_mean"]:
        return "community > global"
    else:
        return "community < global"

merged["comparison_result"] = merged.apply(compare_means, axis=1)

# Reorder and export
merged = merged[["centrality", "community_id", "community_mean", "community_std", "global_mean", "global_std", "comparison_result"]]
merged.to_csv(os.path.join(output_dir, "community_vs_global_centrality_comparison.csv"), index=False)

# Optional plot: same as before
for centrality in merged["centrality"].unique():
    subset = merged[merged["centrality"] == centrality].sort_values("community_id")

    x = np.arange(len(subset))
    width = 0.5

    plt.figure(figsize=(12, 6))
    plt.bar(x, subset["community_mean"], width=width, color="salmon", label="Community Mean")
    plt.axhline(subset["global_mean"].iloc[0], color="green", linestyle="--", linewidth=2, label=f"Global Mean ({subset['global_mean'].iloc[0]:.2f})")

    plt.xticks(x, subset["community_id"], rotation=90)
    plt.ylabel(centrality)
    plt.title(f"{centrality}: Community vs Global Mean")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"{centrality}_mean_comparison.png"))
    plt.close()
