import pandas as pd
import os

# ---------- Paths ----------
input_path = "/home/s3986160/master-thesis/Improvements/community_user_centralities.csv"
output_path = "/home/s3986160/master-thesis/Improvements/community_centrality_stats_by_retention.csv"

# ---------- Load and prepare ----------
df = pd.read_csv(input_path)

# Check expected columns
centrality_cols = ['degree', 'weighted_degree', 'clustering', 'betweenness']

# Melt to long format
df_melted = df.melt(
    id_vars=['retained_25'],
    value_vars=centrality_cols,
    var_name='centrality',
    value_name='value'
)

# Add readable label
df_melted['retention_label'] = df_melted['retained_25'].map({0: 'not_retained', 1: 'retained'})

# Group and aggregate
summary = df_melted.groupby(['centrality', 'retention_label'])['value'].agg(['mean', 'std']).reset_index()

# Save
summary.to_csv(output_path, index=False)
print(f"✅ Saved community centrality stats by retention to: {output_path}")
