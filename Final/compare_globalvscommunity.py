import pandas as pd

# Load files
overall_path = "/home/s3986160/master-thesis/Lastround /output/overall_community_centrality_stats.csv"
community_retention_path = "/home/s3986160/master-thesis/Lastround /output/community_centrality_stats_by_retention.csv"
global_retention_path = "/home/s3986160/master-thesis/Lastround /output/global_retention_centrality_stats.csv"

df_overall = pd.read_csv(overall_path)
df_community = pd.read_csv(community_retention_path)
df_global = pd.read_csv(global_retention_path)

# --- Clean retention labels ---
df_global = df_global[df_global['retention_label'].isin(['0', '1'])].copy()
df_global['retention_label'] = df_global['retention_label'].map({'0': 'not_retained', '1': 'retained'})

# --- PART 1: Community overall vs. Global overall ---
global_avg = df_overall.groupby('centrality').mean().reset_index()

# Merge and keep community_id
comparison_overall = pd.merge(
    df_overall,
    global_avg,
    on='centrality',
    suffixes=('_community', '_global')
)

# Save Part 1
comparison_overall.to_csv("comparison_overall_community_vs_global.csv", index=False)

# --- PART 2: Community retained/not-retained vs. Global retained/not-retained ---
# Keep only centralities that exist in both
shared_centralities = set(df_community['centrality']) & set(df_global['centrality'])
df_community = df_community[df_community['centrality'].isin(shared_centralities)]
df_global = df_global[df_global['centrality'].isin(shared_centralities)]

# Keep community_id in final comparison
comparison_by_retention = pd.merge(
    df_community,
    df_global,
    on=['centrality', 'retention_label'],
    suffixes=('_community', '_global')
)

# Save Part 2
comparison_by_retention.to_csv("comparison_per_community_by_retention_vs_global.csv", index=False)

print("✅ Done: CSVs now include `community_id` where available.")
