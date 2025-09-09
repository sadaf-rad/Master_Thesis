import pandas as pd
import networkx as nx
import os

# === Paths ===
TXN_PATH = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
USER_PATH = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"
RETENTION_PATH = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/retention_AND.csv"

OUT_DIR = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled"
OUTPUT_CSV = f"{OUT_DIR}/weighted_degree_2020.csv"
STATS_CSV = f"{OUT_DIR}/weighted_degree_stats.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# === Load and filter transaction data ===
print("Loading and filtering data...")
df = pd.read_csv(TXN_PATH, on_bad_lines='skip', encoding='utf-8')
user_df = pd.read_csv(USER_PATH, low_memory=False)

# Clean and normalize
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

# Convert timestamps and keep 2020 data
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df_2020 = df[df['timestamp'].dt.year == 2020].copy()

# Keep only users who actually sent transactions
active_senders = set(df_2020['source'])
df_2020 = df_2020[df_2020['source'].isin(active_senders)]

# === Build graph and compute weighted degree ===
print(f"Building graph with {len(df_2020)} edges...")
G = nx.DiGraph()
G.add_weighted_edges_from(zip(df_2020['source'], df_2020['target'], df_2020['weight']))

print("Computing weighted degree centrality...")
weighted_deg = dict(G.degree(weight='weight'))
df_wdeg = pd.DataFrame(weighted_deg.items(), columns=["user", "weighted_degree"])

# === Merge with retention labels ===
df_ret = pd.read_csv(RETENTION_PATH)
df_merged = pd.merge(df_wdeg, df_ret, left_on='user', right_on='address', how='left').drop(columns=['address'])
df_merged.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Weighted degree + retention saved to:\n{OUTPUT_CSV}")

# === Descriptive stats ===
desc = df_wdeg['weighted_degree'].describe().round(6)
desc.to_frame(name='value').to_csv(STATS_CSV)
print(f"📊 Descriptive stats saved to:\n{STATS_CSV}")
