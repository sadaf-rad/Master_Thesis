import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths (20% threshold version)
base_path = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled"
community_stats_path = f"{base_path}/centrality_stats_by_community.csv"
user_level_path = f"{base_path}/community_user_centralities.csv"
global_summary_path = f"{base_path}/global_retention_centrality_stats_20.csv"

# Load data
community_df = pd.read_csv(community_stats_path)
user_df = pd.read_csv(user_level_path)
global_df = pd.read_csv(global_summary_path)
global_df = global_df[global_df['retention_label'] == 'all']

# Centrality name mapping (updated for raw_degree)
centrality_map = {
    "raw_degree": "degree",
    "weighted_degree": "weighted_degree",
    "clustering": "clustering",
    "betweenness": "betweenness"
}

# Extract global means and stds
global_stats = {}
for _, row in global_df.iterrows():
    if row['centrality'] in centrality_map:
        mapped = centrality_map[row['centrality']]
        global_stats[f"{mapped}_mean"] = row['mean']
        global_stats[f"{mapped}_std"] = row['std']

# Step 1: Count retained users per community (20% threshold)
retention_counts = user_df[user_df['retained_20'] == 1].groupby('consensus_community_id').size().rename("retained_count")

# Step 2: Get community size
community_sizes = user_df.groupby('consensus_community_id')['community_size'].first().rename("community_size")

# Step 3: Merge into community_df
community_df = community_df.merge(retention_counts, how='left', left_on='consensus_community_id', right_index=True)
community_df = community_df.merge(community_sizes, how='left', left_on='consensus_community_id', right_index=True)

# Step 4: Clean missing values
community_df['retained_count'] = community_df['retained_count'].fillna(0).astype(int)
community_df['community_size'] = community_df['community_size'].fillna(0).astype(int)

# Step 5: Calculate retention rate
community_df['retention_rate'] = community_df['retained_count'] / community_df['community_size']
community_df['retention_rate'] = community_df['retention_rate'].fillna(0)

# Output directory
output_dir = os.path.join(base_path, "compare")
os.makedirs(output_dir, exist_ok=True)

def plot_centrality(metric, stat_type, global_val):
    col = f"{metric}_{stat_type}"

    # Only keep top 30 largest communities
    top_df = community_df.sort_values("community_size", ascending=False).head(30)

    plt.figure(figsize=(14, 6))

    # Normalize retention rate for color mapping
    min_rate = top_df['retention_rate'].min()
    max_rate = top_df['retention_rate'].max()
    norm = plt.Normalize(min_rate, max_rate)
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    colors = sm.to_rgba(top_df['retention_rate'])

    # Bar plot
    ax = sns.barplot(
        data=top_df,
        x="consensus_community_id",
        y=col,
        palette=colors,
        edgecolor="black"
    )

    # Add dashed red line for global stat
    plt.axhline(global_val, linestyle="--", color="red", linewidth=2, label="Global Mean")

    if metric == "betweenness":
        plt.yscale("log")

    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Retention Rate", rotation=270, labelpad=20)
    tick_values = [min_rate, (min_rate + max_rate) / 2, max_rate]
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.0%}" for v in tick_values])

    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.title(f"{metric.replace('_', ' ').title()} {stat_type.title()} (Top 30 Communities)")
    plt.xlabel("Community ID")
    plt.ylabel(f"{metric.replace('_', ' ').title()} {stat_type.title()}")
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc="upper right")
    plt.tight_layout()

    filename = f"{metric}_{stat_type}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# Generate plots
for metric in ['degree', 'weighted_degree', 'clustering', 'betweenness']:
    for stat_type in ['mean', 'std']:
        key = f"{metric}_{stat_type}"
        if key in global_stats:
            plot_centrality(metric, stat_type, global_stats[key])

print(f"✅ All plots saved in: {output_dir}")
