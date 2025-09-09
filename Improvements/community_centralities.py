import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict

# ------------------- Config -------------------
OUTPUT_DIR = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled"
SARAFU_CSV = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
USER_CSV = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"
COMMUNITY_CSV = f"{OUTPUT_DIR}/leiden_consensus_assignment.csv"
RETENTION_CSV = f"{OUTPUT_DIR}/retention_AND.csv"
THRESHOLD_COL = "retained_20"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- Step 1: Load and Filter Transactions -------------------
print("[1] Loading and filtering transaction data...")
df = pd.read_csv(SARAFU_CSV, on_bad_lines='skip', encoding='utf-8')
users_df = pd.read_csv(USER_CSV, low_memory=False)

df['transfer_subtype'] = df['transfer_subtype'].astype(str).str.upper().str.strip()
df['source'] = df['source'].astype(str).str.strip()
df['target'] = df['target'].astype(str).str.strip()
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])

# Apply filters
df = df[df['transfer_subtype'] == 'STANDARD']
users_df['business_type'] = users_df['business_type'].astype(str).str.upper().str.strip()
users_df['old_POA_blockchain_address'] = users_df['old_POA_blockchain_address'].astype(str).str.strip()
system_accounts = users_df[users_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
df = df[~df['source'].isin(system_accounts)]
df = df[~df['target'].isin(system_accounts)]
df = df[df['timestamp'].dt.year == 2020]

# Keep only active senders
senders_2020 = set(df['source'])
df = df[df['source'].isin(senders_2020)]

# ------------------- Step 2: Build Graph -------------------
print("[2] Building full directed graph...")
G_full = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr='weight', create_using=nx.DiGraph())

# ------------------- Step 3: Load Communities and Retention -------------------
print("[3] Loading community and retention data...")
df_comm = pd.read_csv(COMMUNITY_CSV)
df_ret = pd.read_csv(RETENTION_CSV)[['address', THRESHOLD_COL]]
df_ret.rename(columns={'address': 'user'}, inplace=True)

df_users = pd.merge(df_comm, df_ret, on='user', how='left')

# ------------------- Step 4: Compute Centralities -------------------
print("[4] Computing community-level centralities...")
user_data = []

for comm_id, group in tqdm(df_users.groupby('consensus_community_id'), desc="Communities"):
    members = group['user'].tolist()
    subgraph = G_full.subgraph(members).copy()

    degree = dict(subgraph.degree())
    weighted = dict(subgraph.degree(weight='weight'))
    clustering = nx.clustering(subgraph.to_undirected())

    betweenness = defaultdict(float)
    nodes = list(subgraph.nodes())
    for node in nodes:
        bc = nx.betweenness_centrality_subset(subgraph, sources=[node], targets=nodes, normalized=True)
        for target, value in bc.items():
            betweenness[target] += value

    for user in members:
        user_data.append({
            'user': user,
            'consensus_community_id': comm_id,
            'community_size': len(members),
            'degree': degree.get(user, 0),
            'weighted_degree': weighted.get(user, 0),
            'clustering': clustering.get(user, 0),
            'betweenness': betweenness.get(user, 0)
        })

df_centralities = pd.DataFrame(user_data)
df_centralities = pd.merge(df_centralities, df_ret, on='user', how='left')

# ------------------- Step 5: Save User-Level Centrality File -------------------
print("[5] Saving user-level centrality file...")
df_centralities.to_csv(f"{OUTPUT_DIR}/community_user_centralities.csv", index=False)

# ------------------- Step 6: Community Summary -------------------
print("[6] Saving community structure summary...")
summary_df = pd.DataFrame({
    'total_users_in_communities': [df_centralities['user'].nunique()],
    'total_number_of_communities': [df_centralities['consensus_community_id'].nunique()],
    'average_community_size': [df_centralities['community_size'].mean()]
})
summary_df.to_csv(f"{OUTPUT_DIR}/community_structure_summary.csv", index=False)

# ------------------- Step 7: Per-Community Centrality Stats -------------------
print("[7] Saving per-community centrality stats...")
grouped_stats = df_centralities.groupby('consensus_community_id')[
    ['degree', 'weighted_degree', 'clustering', 'betweenness']
].agg(['mean', 'std'])
grouped_stats.columns = ['_'.join(col) for col in grouped_stats.columns]
grouped_stats.reset_index().to_csv(f"{OUTPUT_DIR}/centrality_stats_by_community.csv", index=False)

print("✅ Done! All community centrality files saved in:", OUTPUT_DIR)
