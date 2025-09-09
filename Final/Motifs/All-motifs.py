import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import time
import os

start_time = time.time()

# === Paths ===
TX_PATH = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
USER_PATH = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"
OUT_PATH = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_details_2020.csv"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# === Step 1: Load & Filter Data ===
print("[Step 1] Loading and filtering transactions...")
df = pd.read_csv(TX_PATH, on_bad_lines='skip', encoding='utf-8')
users_df = pd.read_csv(USER_PATH, low_memory=False)

df['transfer_subtype'] = df['transfer_subtype'].astype(str).str.upper().str.strip()
df['source'] = df['source'].astype(str).str.strip()
df['target'] = df['target'].astype(str).str.strip()
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')

users_df['business_type'] = users_df['business_type'].astype(str).str.upper().str.strip()
users_df['old_POA_blockchain_address'] = users_df['old_POA_blockchain_address'].astype(str).str.strip()

# Filter STANDARD
df = df[df['transfer_subtype'] == 'STANDARD']

# Remove SYSTEM accounts
system_accounts = users_df[users_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
df = df[~df['source'].isin(system_accounts)]
df = df[~df['target'].isin(system_accounts)]

# Filter to 2020 and keep only valid senders
df = df.dropna(subset=['timestamp'])
df_2020 = df[df['timestamp'].dt.year == 2020].copy()
senders_2020 = set(df_2020['source'])
df_2020 = df_2020[df_2020['source'].isin(senders_2020)]

# === Step 2: Build directed graph ===
print("[Step 2] Building directed graph...")
G = nx.DiGraph()
edges = df_2020[['source', 'target']].dropna().values
G.add_edges_from(edges)
print(f"[Info] Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# === Step 3: Find 3-node cycles ===
print("[Step 3] Finding 3-node cycles...")
cycle_user_data = defaultdict(lambda: defaultdict(int))
cycle_global_counts = Counter()

for idx, a in enumerate(G.nodes()):
    if idx % 1000 == 0:
        print(f"  > Checked {idx} users...")
    for b in G.successors(a):
        for c in G.successors(b):
            if G.has_edge(c, a):  # A → B → C → A
                nodes = tuple(sorted([a, b, c]))
                cycle_global_counts[nodes] += 1
                for user in [a, b, c]:
                    peers = tuple(sorted(set([a, b, c]) - {user}))
                    cycle_user_data[user][peers] += 1

# === Step 4: Find reciprocals ===
print("[Step 4] Finding reciprocals...")
recip_user_data = defaultdict(lambda: defaultdict(int))
recip_global_counts = Counter()
seen = set()

for u, v in G.edges():
    if G.has_edge(v, u) and (v, u) not in seen:
        pair = tuple(sorted([u, v]))
        recip_global_counts[pair] += 1
        recip_user_data[u][(v,)] += 1
        recip_user_data[v][(u,)] += 1
        seen.add((u, v))

# === Step 5: Save Results ===
print("[Step 5] Saving results...")
rows = []

for user, peer_dict in cycle_user_data.items():
    for peers, count in peer_dict.items():
        key = tuple(sorted([user, *peers]))
        rows.append({
            "user": user,
            "motif_type": "cycle_3node",
            "peers": peers,
            "user_count": count,
            "global_count": cycle_global_counts[key]
        })

for user, peer_dict in recip_user_data.items():
    for peers, count in peer_dict.items():
        key = tuple(sorted([user, *peers]))
        rows.append({
            "user": user,
            "motif_type": "reciprocal",
            "peers": peers,
            "user_count": count,
            "global_count": recip_global_counts[key]
        })

motif_df = pd.DataFrame(rows)
motif_df.to_csv(OUT_PATH, index=False)

print(f"[Done] Saved to: {OUT_PATH}")
print("--- Total runtime: %.2f minutes ---" % ((time.time() - start_time) / 60))
