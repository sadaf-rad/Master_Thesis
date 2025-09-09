import pandas as pd
import networkx as nx
import os

# === Paths ===
TXN_PATH = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
USER_PATH = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"
RETENTION_PATH = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/retention_AND.csv"

OUT_DIR = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled"
CENTRALITY_CSV = f"{OUT_DIR}/degree_centrality_2020.csv"
DESC_CSV = f"{OUT_DIR}/degree_centrality_stats.csv"

os.makedirs(OUT_DIR, exist_ok=True),

# === Load data ===
print("Loading and filtering transaction data...")
df = pd.read_csv(TXN_PATH, on_bad_lines='skip', encoding='utf-8')
user_df = pd.read_csv(USER_PATH, low_memory=False)

# Clean columns
df['transfer_subtype'] = df['transfer_subtype'].astype(str).str.upper().str.strip()
df['source'] = df['source'].astype(str).str.strip()
df['target'] = df['target'].astype(str).str.strip()
user_df['business_type'] = user_df['business_type'].astype(str).str.upper().str.strip()
user_df['old_POA_blockchain_address'] = user_df['old_POA_blockchain_address'].astype(str).str.strip()

# Filter standard transactions and remove system accounts
df = df[df['transfer_subtype'] == 'STANDARD']
system_accounts = user_df[user_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
df = df[~df['source'].isin(system_accounts)]
df = df[~df['target'].isin(system_accounts)]

# Keep only 2020 transactions with valid timestamps
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df_2020 = df[df['timestamp'].dt.year == 2020].copy()

# Keep only active senders
senders_2020 = set(df_2020['source'])
df_2020 = df_2020[df_2020['source'].isin(senders_2020)]

# === Build graph and compute centrality ===
print(f"Building graph with {len(df_2020)} edges...")
G = nx.DiGraph()
G.add_edges_from(zip(df_2020['source'], df_2020['target']))
G = G.subgraph(senders_2020).copy()

print(f"Computing normalized degree centrality for {G.number_of_nodes()} users...")
deg_cent = nx.degree_centrality(G)

# Save per-user centrality
df_deg = pd.DataFrame(deg_cent.items(), columns=["user", "degree_centrality"])

# Merge with retention labels (all thresholds)
df_ret = pd.read_csv(RETENTION_PATH)
df_merged = pd.merge(df_deg, df_ret, left_on='user', right_on='address', how='left').drop(columns=['address'])
df_merged.to_csv(CENTRALITY_CSV, index=False)
print(f"✅ Degree centrality + retention saved to:\n{CENTRALITY_CSV}")

# Compute and save descriptive statistics
desc = df_deg['degree_centrality'].describe().round(6)
desc.to_frame(name='value').to_csv(DESC_CSV)
print(f"📊 Descriptive stats saved to:\n{DESC_CSV}")
