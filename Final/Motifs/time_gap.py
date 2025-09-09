import pandas as pd
from collections import defaultdict
import numpy as np
import os

print("[Step 1] Loading and filtering data...")
motif_file = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_details_2020.csv"
tx_file = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
user_file = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"

motifs = pd.read_csv(motif_file)
txns = pd.read_csv(tx_file, on_bad_lines='skip', parse_dates=["timeset"], encoding='utf-8')
users_df = pd.read_csv(user_file, low_memory=False)

# Normalize fields
txns['transfer_subtype'] = txns['transfer_subtype'].astype(str).str.upper().str.strip()
txns['source'] = txns['source'].astype(str).str.strip()
txns['target'] = txns['target'].astype(str).str.strip()
users_df['business_type'] = users_df['business_type'].astype(str).str.upper().str.strip()
users_df['old_POA_blockchain_address'] = users_df['old_POA_blockchain_address'].astype(str).str.strip()

# Filter: standard only
txns = txns[txns['transfer_subtype'] == 'STANDARD']

# Remove system accounts
system_users = users_df[users_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
txns = txns[~txns['source'].isin(system_users)]
txns = txns[~txns['target'].isin(system_users)]

# Year filter
txns = txns.dropna(subset=['timeset'])
txns = txns[txns['timeset'].dt.year == 2020]

# Keep only senders
senders_2020 = set(txns['source'])
txns = txns[txns['source'].isin(senders_2020)]

print("[Step 2] Creating edge-time lookup...")
edge_times = defaultdict(list)
for _, row in txns.iterrows():
    edge_times[(row['source'], row['target'])].append(row['timeset'])

for k in edge_times:
    edge_times[k] = sorted(edge_times[k])

print("[Step 3] Computing time gaps...")
results = []

for _, row in motifs.iterrows():
    user = row['user']
    motif_type = row['motif_type']
    peers = eval(row['peers']) if isinstance(row['peers'], str) else row['peers']
    
    time_deltas = []

    if motif_type == "cycle_3node":
        if len(peers) != 2:
            continue
        nodes = [user] + list(peers)
        nodes = sorted(nodes)

        edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])]
        times = []
        for u, v in edges:
            if (u, v) in edge_times and edge_times[(u, v)]:
                times.append(edge_times[(u, v)][0])
            else:
                break
        if len(times) == 3:
            times = sorted(times)
            delta1 = (times[1] - times[0]).total_seconds()
            delta2 = (times[2] - times[1]).total_seconds()
            avg_gap = np.mean([delta1, delta2])
            results.append({
                "user": user,
                "motif_type": motif_type,
                "peers": peers,
                "avg_time_gap_sec": avg_gap
            })

    elif motif_type == "reciprocal":
        if len(peers) != 1:
            continue
        peer = peers[0]
        pair1 = (user, peer)
        pair2 = (peer, user)
        if pair1 in edge_times and pair2 in edge_times:
            for t1 in edge_times[pair1]:
                next_t2 = [t2 for t2 in edge_times[pair2] if t2 > t1]
                if next_t2:
                    gap = (next_t2[0] - t1).total_seconds()
                    time_deltas.append(gap)
                    break
            if time_deltas:
                avg_gap = np.mean(time_deltas)
                results.append({
                    "user": user,
                    "motif_type": motif_type,
                    "peers": peers,
                    "avg_time_gap_sec": avg_gap
                })

print("\n[Summary] Average time gap by motif type:")
summary_df = pd.DataFrame(results)

if not summary_df.empty:
    for motif in summary_df['motif_type'].unique():
        motif_avg = summary_df[summary_df['motif_type'] == motif]['avg_time_gap_sec'].mean()
        hours = motif_avg / 3600
        days = motif_avg / (3600 * 24)
        print(f"  - {motif}: {motif_avg:.2f} sec ≈ {hours:.2f} hours ≈ {days:.2f} days")
else:
    print("  ⚠️ No time gaps were computed. Check motif/transaction data.")

print("\n[Step 5] Saving time gap results...")
timegap_df = pd.DataFrame(results)
out_path = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_timegaps_2020.csv"
timegap_df.to_csv(out_path, index=False)
print(f"[Done] Saved to: {out_path}")
