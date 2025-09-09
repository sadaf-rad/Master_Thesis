import pandas as pd
import networkx as nx
from tqdm import tqdm
import time
import os

# === Paths ===
TXN_PATH = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
USER_PATH = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"
RETENTION_PATH = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/retention_AND.csv"
OUT_DIR = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_CSV = f"{OUT_DIR}/betweenness_centrality_2020.csv"
DESC_CSV = f"{OUT_DIR}/betweenness_centrality_stats.csv"

print("Loading and filtering data...")
df = pd.read_csv(TXN_PATH, on_bad_lines='skip', encoding='utf-8')
user_df = pd.read_csv(USER_PATH, low_memory=False)

df['transfer_subtype'] = df['transfer_subtype'].astype(str).str.upper().str.strip()
df['source'] = df['source'].astype(str).str.strip()
df['target'] = df['target'].astype(str).str.strip()
user_df['business_type'] = user_df['business_type'].astype(str).str.upper().str.strip()
user_df['old_POA_blockchain_address'] = user_df['old_POA_blockchain_address'].astype(str).str.strip()

df = df[df['transfer_subtype'] == 'STANDARD']
system_accounts = user_df[user_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
df = df[~df['source'].isin(system_accounts)]
df = df[~df['target'].isin(system_accounts)]

df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df_2020 = df[df['timestamp'].dt.year == 2020].copy()

active_senders = set(df_2020['source'])
df_2020 = df_2020[df_2020['source'].isin(active_senders)]

print(f"Building directed graph with {len(df_2020)} edges...")
G = nx.DiGraph()
G.add_edges_from(zip(df_2020['source'], df_2020['target']))
G = G.subgraph(active_senders).copy()

print("Calculating betweenness centrality...")
start_time = time.time()

betweenness = dict()
nodes = list(G.nodes())

for i, node in enumerate(tqdm(nodes, desc="Processing nodes", unit="node")):
    centrality = nx.betweenness_centrality_subset(G, sources=[node], targets=nodes, normalized=True)
    for k, v in centrality.items():
        betweenness[k] = betweenness.get(k, 0) + v

end_time = time.time()
print(f"Finished in {round(end_time - start_time, 2)} seconds.")

# Save results with retention
df_bet = pd.DataFrame({
    'user': list(betweenness.keys()),
    'betweenness_centrality': list(betweenness.values())
})
df_ret = pd.read_csv(RETENTION_PATH)
df_final = pd.merge(df_bet, df_ret, left_on='user', right_on='address', how='left').drop(columns=['address'])
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved to {OUTPUT_CSV}")

# Save descriptive statistics
df_bet['betweenness_centrality'].describe().round(6).to_frame(name="value").to_csv(DESC_CSV)
print(f"📊 Descriptive stats saved to {DESC_CSV}")
