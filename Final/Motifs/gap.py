import pandas as pd
import numpy as np
from collections import defaultdict


motif_file = "/home/s3986160/master-thesis/Results/user_motif_details_2020.csv"
tx_file = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
retention_file = "/home/s3986160/master-thesis/Retention/new definition/retention_AND.csv"
output_file = "/home/s3986160/master-thesis/Phase 3/Motifs/Results/motif_gap_retention_summary.csv"

motifs = pd.read_csv(motif_file)
txns = pd.read_csv(tx_file, parse_dates=["timeset"])
txns = txns[txns['timeset'].dt.year == 2020]
retention = pd.read_csv(retention_file)
retention = retention[['address', 'retained_25']].rename(columns={'address': 'user', 'retained_25': 'retained'})

edge_times = defaultdict(list)
for _, row in txns.iterrows():
    edge_times[(row['source'], row['target'])].append(row['timeset'])
for k in edge_times:
    edge_times[k] = sorted(edge_times[k])

motif_records = []
processed_cycles = set()
processed_reciprocals = set()

for _, row in motifs.iterrows():
    user = row['user']
    motif_type = row['motif_type']
    peers = eval(row['peers']) if isinstance(row['peers'], str) else row['peers']

    if motif_type == "cycle_3node" and len(peers) == 2:
        nodes = tuple(sorted([user] + list(peers)))
        if nodes in processed_cycles:
            continue
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
            avg_gap_days = np.mean([delta1, delta2]) / (3600 * 24)
            users_in_motif = list(nodes)
            retained_labels = retention[retention['user'].isin(users_in_motif)]['retained']
            retained_count = retained_labels.sum()
            not_retained_count = len(users_in_motif) - retained_count
            motif_records.append({
                "motif_type": "cycle_3node",
                "users": users_in_motif,
                "avg_gap_days": avg_gap_days,
                "retained_count": int(retained_count),
                "not_retained_count": int(not_retained_count),
                "retained_ratio": retained_count / len(users_in_motif)
            })
            processed_cycles.add(nodes)

    elif motif_type == "reciprocal" and len(peers) == 1:
        peer = peers[0]
        pair = tuple(sorted([user, peer]))
        if pair in processed_reciprocals:
            continue
        if (user, peer) in edge_times and (peer, user) in edge_times:
            for t1 in edge_times[(user, peer)]:
                t2s = [t2 for t2 in edge_times[(peer, user)] if t2 > t1]
                if t2s:
                    delta = (t2s[0] - t1).total_seconds() / (3600 * 24)
                    users_in_motif = list(pair)
                    retained_labels = retention[retention['user'].isin(users_in_motif)]['retained']
                    retained_count = retained_labels.sum()
                    not_retained_count = len(users_in_motif) - retained_count
                    motif_records.append({
                        "motif_type": "reciprocal",
                        "users": users_in_motif,
                        "avg_gap_days": delta,
                        "retained_count": int(retained_count),
                        "not_retained_count": int(not_retained_count),
                        "retained_ratio": retained_count / len(users_in_motif)
                    })
                    break
        processed_reciprocals.add(pair)

motif_df = pd.DataFrame(motif_records)
motif_df.to_csv(output_file, index=False)
print(f"[Done] Saved to:\n  {output_file}")
