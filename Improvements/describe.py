import pandas as pd

# === Step 1: Load and clean ===
def load_and_clean(path, rename_col=None):
    df = pd.read_csv(path)
    if rename_col:
        df.rename(columns=rename_col, inplace=True)
    if 'address' in df.columns:
        df['address'] = df['address'].astype(str).str.lower().str.strip()
    return df

print("[1] Loading files...")

df_retention = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/retention_AND.csv")[['address', 'retained_25']]
df_retention['retained_25'] = df_retention['retained_25'].astype(int)

df_degree = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/raw_degree_2020.csv", rename_col={'user': 'address'})

df_clustering = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/clustering_coefficients_2020.csv")
df_clustering.rename(columns={'clustering_coefficient': 'clustering'}, inplace=True)

df_betweenness = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/betweenness_centrality_2020.csv")
df_betweenness.rename(columns={'betweenness_centrality': 'betweenness'}, inplace=True)

df_cyclic = load_and_clean("/home/s3986160/master-thesis/Retention/new definition/Centralities 2020/cyclic_vs_retention_25.csv")[['address', 'cyclic_status']]
df_cyclic['cyclic_status'] = df_cyclic['cyclic_status'].map({'acyclic': 0, 'cyclic': 1})

motif_counts = pd.read_csv("/home/s3986160/master-thesis/Results/user_motif_details_2020.csv")
motif_counts.rename(columns={"user": "address"}, inplace=True)
motif_counts['address'] = motif_counts['address'].str.lower().str.strip()
motif_counts_pivot = motif_counts.pivot_table(index='address', columns='motif_type', values='user_count', aggfunc='sum')
motif_counts_pivot.columns = [f'motif_count_{col}' for col in motif_counts_pivot.columns]
motif_counts_pivot.reset_index(inplace=True)

motif_gaps = pd.read_csv("/home/s3986160/master-thesis/Results/user_motif_timegaps_2020.csv")
motif_gaps.rename(columns={"user": "address"}, inplace=True)
motif_gaps['address'] = motif_gaps['address'].str.lower().str.strip()
motif_gaps_pivot = motif_gaps.pivot_table(index='address', columns='motif_type', values='avg_time_gap_sec', aggfunc='mean')
motif_gaps_pivot.columns = [f'avg_gap_{col}' for col in motif_gaps_pivot.columns]
motif_gaps_pivot.reset_index(inplace=True)

df_demo = pd.read_csv("/home/s3986160/master-thesis/Results/global_users_demographics_2020_2021.csv")
df_demo['address'] = df_demo['address'].str.lower().str.strip()
df_demo = df_demo[['address', 'gender', 'area_name', 'business_type']]
df_demo[['gender', 'area_name', 'business_type']] = df_demo[['gender', 'area_name', 'business_type']].fillna('Unknown')

# === Step 2: Merge and generate user descriptions ===
print("[2] Merging features and generating user descriptions...")

df = df_retention.copy()
df = df.merge(df_degree, on='address', how='left')
df = df.merge(df_clustering, on='address', how='left')
df = df.merge(df_betweenness, on='address', how='left')
df = df.merge(df_cyclic, on='address', how='left')
df = df.merge(motif_counts_pivot, on='address', how='left')
df = df.merge(motif_gaps_pivot, on='address', how='left')
df = df.merge(df_demo, on='address', how='left')

# Fill missing values for text-friendly features
df[['area_name', 'gender', 'business_type']] = df[['area_name', 'gender', 'business_type']].fillna('Unknown')
df[['degree', 'clustering', 'betweenness']] = df[['degree', 'clustering', 'betweenness']].fillna(0)
if 'motif_count_3cycle' not in df.columns:
    df['motif_count_3cycle'] = 0
if 'avg_gap_reciprocal' not in df.columns:
    df['avg_gap_reciprocal'] = 0

# Function to create the sentence
def describe_user(row):
    return (
        f"The user is from {row['area_name']}, is {row['gender']}, works in {row['business_type']}, "
        f"has a degree of {row['degree']:.1f}, clustering coefficient of {row['clustering']:.2f}, "
        f"and betweenness centrality of {row['betweenness']:.4f}. "
        f"They are labeled as {'cyclic' if row['cyclic_status'] == 1 else 'acyclic'}, "
        f"appeared in {int(row['motif_count_3cycle'])} 3-node motifs, "
        f"and completed reciprocal motifs in {row['avg_gap_reciprocal']:.1f} days on average."
    )

# Apply to all users
df['user_description'] = df.apply(describe_user, axis=1)

# === Step 3: Save the result ===
print("[3] Saving user descriptions...")
df[['address', 'user_description']].to_csv("/home/s3986160/master-thesis/Results/user_descriptions_2020.csv", index=False)
print("✅ Done! Descriptions saved to Results/user_descriptions_2020.csv")
