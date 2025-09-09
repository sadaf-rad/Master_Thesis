import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
import matplotlib.colors as mcolors

# Use safe backend
matplotlib.use('Agg')

# === File Paths ===
centrality_summary_path = "/home/s3986160/master-thesis/Lastround /output/community_vs_global_centrality_comparison.csv"
user_level_path = "/home/s3986160/master-thesis/Lastround /output/community_users_with_all_centralities.csv"
output_dir = "/home/s3986160/master-thesis/Lastround /output/"
os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
df_centrality = pd.read_csv(centrality_summary_path)
df_users = pd.read_csv(user_level_path)

# === Fix dtypes ===
df_users["community_id"] = df_users["community_id"].astype(float)
df_centrality["community_id"] = df_centrality["community_id"].astype(float)

# === Count Retained Users Per Community ===
df_users["retained_flag"] = df_users["retention_label"].apply(lambda x: 1 if str(x).strip().lower() == "retained" else 0)
retention_stats = df_users.groupby("community_id")["retained_flag"].sum().reset_index()
retention_stats.columns = ["community_id", "retention_count"]

# === Merge Retention Count into Centrality Summary ===
df_plot = df_centrality.merge(retention_stats, on="community_id", how="left")
df_plot["retention_count"] = df_plot["retention_count"].fillna(0)

# === Create Plots for Mean and Std ===
for value_type in ["mean", "std"]:
    value_col = f"community_{value_type}"
    global_col = f"global_{value_type}"

    for centrality in df_plot["centrality"].unique():
        sub = df_plot[df_plot["centrality"] == centrality].sort_values("community_id").copy()
        if sub.empty or value_col not in sub.columns or global_col not in sub.columns:
            continue

        x = np.arange(len(sub))
        global_val = sub[global_col].iloc[0]

        # === Log-scaled Color Mapping for Retention Count ===
        cmap = plt.cm.plasma  # vivid and distinct
        norm = mcolors.LogNorm(vmin=max(sub["retention_count"].min(), 1), vmax=sub["retention_count"].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        colors = [(*rgba[:3], 1.0) for rgba in sm.to_rgba(sub["retention_count"])]

        # === Plot ===
        plt.figure(figsize=(14, 6))
        bars = plt.bar(x, sub[value_col], color=colors, edgecolor="black", label=f"Community {value_type.title()}")
        plt.axhline(global_val, color="green", linestyle="--", linewidth=2,
                    label=f"Global {value_type.title()} = {global_val:.2e}")

        # Add retention count labels
        max_height = sub[value_col].max()
        for i, count in enumerate(sub["retention_count"]):
            y_val = sub[value_col].iloc[i]
            if pd.notna(y_val) and np.isfinite(y_val) and y_val < max_height * 1.5:
                plt.text(i, y_val * 1.01, str(int(count)), ha='center', va='bottom', fontsize=8)

        # Axes and layout
        plt.xticks(x, sub["community_id"].astype(int), rotation=90)
        plt.ylabel(f"{centrality} ({value_type})")
        plt.title(f"{centrality.capitalize()} per Community - ({value_type})")

        cbar = plt.colorbar(sm)
        cbar.set_label("Retention Count ")

        plt.legend()
        plt.tight_layout()

        filename = f"{centrality}_{value_type}_centrality_retention_colored.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
