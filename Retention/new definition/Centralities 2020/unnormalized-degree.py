import pandas as pd
import networkx as nx
import os

# === Paths ===
TXN_PATH = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
USER_PATH = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"
OUTPUT_FILE = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/raw_degree_2020.csv"

# === Load transaction and user data ===
print("Loading data...")
df = pd.read_csv(TXN_PATH, on_bad_lines='skip', encoding='utf-8')
user_df = pd.read_csv(USER_PATH, low_memory=False)

# === Clean columns ===
df['transfer_subtype'] = df['transfer_subtype'].astype(str).str.upper().str.strip()
df['source'] = df['source'].astype(str).str.strip()
df['target'] = df['target'].astype(str).str.strip()
user_df['business_type'] = user_df['business_type'].astype(str).str.upper().str.strip()
user_df['old_POA_blockchain_address'] = user_df['old_POA_blockchain_address'].astype(str).str.strip()

# === Filter to standard transactions only ===
df = df[df['transfer_subtype'] == 'STANDARD']

# === Remove SYSTEM accounts ===
system_accounts = user_df[user_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
df = df[~df['source'].isin(system_accounts)]
df = df[~df['target'].isin(system_accounts)]

# === Filter to 2020 ===
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df_2020 = df[df['timestamp'].dt.year == 2020].copy()

# === Keep only active senders ===
active_senders = set(df_2020['source'])
df_2020 = df_2020[df_2020['source'].isin(active_senders)]

# === Build graph and compute raw degree ===
print(f"Building graph from {len(df_2020)} edges...")
G = nx.DiGraph()
G.add_edges_from(zip(df_2020['source'], df_2020['target']))
G = G.subgraph(active_senders).copy()

print(f"Computing raw (total) degree for {G.number_of_nodes()} users...")
raw_degree = dict(G.degree())  # total degree (in + out)

# === Save output ===
out_df = pd.DataFrame(raw_degree.items(), columns=["user", "raw_degree"])
out_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved to: {OUTPUT_FILE}")
