import pandas as pd
import networkx as nx
import os

# === Paths ===
TXN_PATH = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
USER_PATH = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"
RETENTION_PATH = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/retention_AND.csv"

OUTPUT_DIR = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled"
CLUSTERING_CSV = f"{OUTPUT_DIR}/global_user_clustering.csv"
DESC_CSV = f"{OUTPUT_DIR}/global_clustering_stats.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Step 1: Load and preprocess transaction data ===
print("Loading and filtering transaction data...")
df = pd.read_csv(TXN_PATH, on_bad_lines='skip', encoding='utf-8')
df['transfer_subtype'] = df['transfer_subtype'].astype(str).str.upper().str.strip()
df['source'] = df['source'].astype(str).str.strip()
df['target'] = df['target'].astype(str).str.strip()

# Filter: only STANDARD transactions
df = df[df['transfer_subtype'] == 'STANDARD']

# Remove system accounts
users_df = pd.read_csv(USER_PATH, low_memory=False)
users_df['business_type'] = users_df['business_type'].astype(str).str.upper().str.strip()
users_df['old_POA_blockchain_address'] = users_df['old_POA_blockchain_address'].astype(str).str.strip()
system_accounts = users_df[users_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
df = df[~df['source'].isin(system_accounts)]
df = df[~df['target'].isin(system_accounts)]

# Filter to 2020
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df_2020 = df[df['timestamp'].dt.year == 2020].copy()

# Keep only users who actually sent transactions
active_senders = set(df_2020['source'])
df_2020 = df_2020[df_2020['source'].isin(active_senders)]

# === Step 2: Build graph and compute clustering ===
print("Building graph and computing global clustering coefficient...")
G = nx.from_pandas_edgelist(df_2020, 'source', 'target', create_using=nx.DiGraph())
G = G.subgraph(active_senders).copy()

# Compute clustering coefficient (undirected version)
clustering = nx.clustering(G.to_undirected())
df_clustering = pd.DataFrame(list(clustering.items()), columns=['user', 'global_clustering'])

# === Step 3: Merge with retention thresholds ===
print("Merging with retention labels...")
df_retention = pd.read_csv(RETENTION_PATH)
df_final = pd.merge(df_clustering, df_retention, left_on='user', right_on='address', how='left').drop(columns=['address'])

# Save per-user clustering + retention
df_final.to_csv(CLUSTERING_CSV, index=False)
print(f"✅ Saved: {CLUSTERING_CSV}")

# === Step 4: Descriptive statistics ===
print("Computing and saving descriptive stats...")
desc = df_clustering['global_clustering'].describe().round(6)
desc.to_frame(name='value').to_csv(DESC_CSV)
print(f"📊 Saved: {DESC_CSV}")
