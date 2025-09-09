import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# === Paths ===
TX_PATH = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
USER_PATH = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"
MOTIF_PATH = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_details_2020.csv"
OUT_DIR = "/home/s3986160/master-thesis/Plots"
os.makedirs(OUT_DIR, exist_ok=True)

# === Step 1: Load and clean transactions ===
print("[Step 1] Loading data...")
txns = pd.read_csv(TX_PATH, on_bad_lines='skip', encoding='utf-8', parse_dates=['timeset'])
users_df = pd.read_csv(USER_PATH, low_memory=False)
motifs = pd.read_csv(MOTIF_PATH)

txns['transfer_subtype'] = txns['transfer_subtype'].astype(str).str.upper().str.strip()
txns['source'] = txns['source'].astype(str).str.strip()
txns['target'] = txns['target'].astype(str).str.strip()
users_df['business_type'] = users_df['business_type'].astype(str).str.upper().str.strip()
users_df['old_POA_blockchain_address'] = users_df['old_POA_blockchain_address'].astype(str).str.strip()

# Filter standard and remove system users
txns = txns[txns['transfer_subtype'] == 'STANDARD']
system_users = users_df[users_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
txns = txns[~txns['source'].isin(system_users)]
txns = txns[~txns['target'].isin(system_users)]

# Filter to 2020 with valid timestamps
txns = txns.dropna(subset=['timeset'])
txns = txns[txns['timeset'].dt.year == 2020].copy()

# === Step 2: Create edge-time lookup ===
print("[Step 2] Creating edge-time lookup...")
edge_times = defaultdict(list)
for _, row in txns.iterrows():
    edge_times[(row['source'], row['target'])].append(row['timeset'])

for k in edge_times:
    edge_times[k] = sorted(edge_times[k])

# === Step 3: Assign timestamps to motifs ===
print("[Step 3] Timestamping motif instances...")
motif_timestamps = []

for _, row in motifs.iterrows():
    user = row['user']
    motif_type = row['motif_type']
    peers = eval(row['peers']) if isinstance(row['peers'], str) else row['peers']
    timestamp = None

    if motif_type == "cycle_3node" and len(peers) == 2:
        nodes = sorted([user, *peers])
        edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])]
        times = []
        for u, v in edges:
            if (u, v) in edge_times and edge_times[(u, v)]:
                times.append(edge_times[(u, v)][0])
            else:
                break
        if len(times) == 3:
            timestamp = min(times)

    elif motif_type == "reciprocal" and len(peers) == 1:
        peer = peers[0]
        pair1 = (user, peer)
        pair2 = (peer, user)
        t1s = edge_times.get(pair1, [])
        t2s = edge_times.get(pair2, [])
        if t1s and t2s:
            t1 = t1s[0]
            t2_candidates = [t for t in t2s if t > t1]
            if t2_candidates:
                timestamp = t1

    if timestamp:
        motif_timestamps.append({
            "motif_type": motif_type,
            "timestamp": timestamp
        })

motif_df = pd.DataFrame(motif_timestamps)
motif_df['month'] = motif_df['timestamp'].dt.to_period('M')

# === Step 4: Aggregate monthly motif counts ===
monthly_motifs = motif_df.groupby(['month', 'motif_type']).size().unstack(fill_value=0)

# === Step 5: Aggregate monthly transaction counts ===
txns['month'] = txns['timeset'].dt.to_period('M')
monthly_txns = txns.groupby('month').size()

# === Step 6: Plot ===
print("[Step 6] Plotting...")
fig, ax1 = plt.subplots(figsize=(14, 6))

# Area plot for motifs
monthly_motifs.index = monthly_motifs.index.to_timestamp()
monthly_motifs.plot.area(ax=ax1, alpha=0.6)
ax1.set_ylabel("Motif counts")
ax1.set_xlabel("Month")
ax1.legend(title="Motif type", loc='upper left')

# Overlay transactions (right axis)
ax2 = ax1.twinx()
ax2.plot(monthly_txns.index.to_timestamp(), monthly_txns.values, 'r--', label='Transactions')
ax2.set_ylabel("Number of transactions", color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Monthly motif counts and transactions in Sarafu (2020)")
plt.tight_layout()

# Save
plot_path = f"{OUT_DIR}/newmotif_vs_txn_trends_2020.png"
plt.savefig(plot_path, dpi=300)
print(f"[Done] Plot saved at: {plot_path}")
