import pandas as pd

# Load the data
file_path = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/community_user_centralities.csv"
df = pd.read_csv(file_path)

# Define the centrality columns
centrality_columns = ['degree', 'weighted_degree', 'clustering', 'betweenness']

# Split the dataframe based on retention label
retained = df[df['retained_20'] == 1]
not_retained = df[df['retained_20'] == 0]

# Prepare the summary dictionary
summary = {}

for col in centrality_columns:
    retained_mean = retained[col].mean()
    retained_std = retained[col].std()
    not_retained_mean = not_retained[col].mean()
    not_retained_std = not_retained[col].std()

    summary[col] = {
        'Not Retained (mean ± std)': f"{not_retained_mean:.2f} ± {not_retained_std:.2f}",
        'Retained (mean ± std)': f"{retained_mean:.2f} ± {retained_std:.2f}"
    }

# Convert to DataFrame
summary_df = pd.DataFrame(summary).T
summary_df.index.name = 'Centrality Metric'

# Save to CSV
output_path = "/home/s3986160/master-thesis/Results/retained_vs_notretained_community_centralities.csv"
summary_df.to_csv(output_path)

print(f"Summary saved to: {output_path}")
