import pandas as pd
import os

# Load the dataset
file_path = "/home/s3986160/master-thesis/Lastround /output/community_users_with_all_centralities.csv"
df = pd.read_csv(file_path)

# Define the centrality columns
centrality_cols = ["degree", "betweenness", "closeness", "clustering", "kshell", "weighted_degree"]

# Prepare output directory
output_dir = "/home/s3986160/master-thesis/Phase 3/output"
os.makedirs(output_dir, exist_ok=True)

# Prepare a list to collect overall stats per community
overall_stats = []

# Prepare a list to collect stats by retention label per community
label_stats = []

# Group by community
for community_id, group in df.groupby("community_id"):
    for centrality in centrality_cols:
        # General descriptive stats for this centrality in this community
        desc = group[centrality].describe()
        desc["community_id"] = community_id
        desc["centrality"] = centrality
        overall_stats.append(desc)

        # Split by retention label
        for label, label_group in group.groupby("retention_label"):
            label_desc = label_group[centrality].describe()
            label_desc["community_id"] = community_id
            label_desc["centrality"] = centrality
            label_desc["retention_label"] = label
            label_stats.append(label_desc)

# Convert lists to DataFrames
overall_stats_df = pd.DataFrame(overall_stats)
label_stats_df = pd.DataFrame(label_stats)

# Save to CSV
overall_path = os.path.join(output_dir, "overall_community_centrality_stats.csv")
label_path = os.path.join(output_dir, "community_centrality_stats_by_retention.csv")

overall_stats_df.to_csv(overall_path, index=False)
label_stats_df.to_csv(label_path, index=False)

## Print all rows without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("Overall centrality stats per community:")
print(overall_stats_df)

print("\nCentrality stats per community split by retention label:")
print(label_stats_df)
