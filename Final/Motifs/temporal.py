import pandas as pd
import networkx as nx
from collections import defaultdict
from datetime import timedelta
import os

# === CONFIG ===
TX_PATH = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
OUT_PATH = "/home/s3986160/master-thesis/Results/user_motif_details_temporal.csv"
WINDOW_SIZE_DAYS = 60
STEP_SIZE_DAYS = 30

# === Load Data ===
print("[1] Loading transaction data...")
df = pd.read_csv(TX_PATH)
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])

df = df[['source', 'target', 'timestamp']].dropna()
df = df[df['timestamp'].dt.year == 2020]

# === Create rolling windows ===
start_time = df['timestamp'].min().normalize()
end_time = df['timestamp'].max().normalize()

windows = []
current_start = start_time

print("[2] Creating rolling time windows...")
while current_start < end_time:
    current_end = current_start + timedelta(days=WINDOW_SIZE_DAYS)
    windows.append((current_start, current_end))
    current_start += timedelta(days=STEP_SIZE_DAYS)

print(f"  > Total windows: {len(windows)}")

motif_records = []

# === Motif Detection per Window ===
for i, (win_start, win_end) in enumerate(windows):
    print(f"[3] Processing window {i+1}/{len(windows)}: {win_start.date()} to {win_end.date()}")

    df_window = df[(df['timestamp'] >= win_start) & (df['timestamp'] < win_end)]
    G = nx.DiGraph()
    G.add_edges_from(zip(df_window['source'], df_window['target']))

    # Build edge timestamp lookup
    edge_times = defaultdict(list)
    for _, row in df_window.iterrows():
        edge_times[(row['source'], row['target'])].append(row['timestamp'])

    # Reciprocals
    seen_pairs = set()
    for u, v in G.edges():
        if G.has_edge(v, u) and (v, u) not in seen_pairs:
            t1s = edge_times.get((u, v), [])
            t2s = edge_times.get((v, u), [])
            if t1s and t2s:
                timestamp = max(min(t1s), min(t2s))
                motif_records.append({
                    "user": u,
                    "motif_type": "reciprocal",
                    "peers": [v],
                    "timestamp": timestamp,
                    "window_start": win_start,
                    "window_end": win_end
                })
            seen_pairs.add((u, v))

    # 3-node cycles
    seen_cycles = set()
    for a in G.nodes():
        for b in G.successors(a):
            for c in G.successors(b):
                if G.has_edge(c, a):
                    cycle_nodes = tuple(sorted([a, b, c]))
                    if cycle_nodes in seen_cycles:
                        continue
                    seen_cycles.add(cycle_nodes)

                    edges = [(a, b), (b, c), (c, a)]
                    edge_ts = [min(edge_times[e]) for e in edges if e in edge_times]
                    if len(edge_ts) == 3:
                        motif_time = max(edge_ts)  # Completion time
                        for user in [a, b, c]:
                            peers = tuple(sorted(set([a, b, c]) - {user}))
                            motif_records.append({
                                "user": user,
                                "motif_type": "cycle_3node",
                                "peers": peers,
                                "timestamp": motif_time,
                                "window_start": win_start,
                                "window_end": win_end
                            })

# === Save Output ===
print("[4] Saving motif timeline...")
motif_df = pd.DataFrame(motif_records)
motif_df.to_csv(OUT_PATH, index=False)
print(f"[✅ Done] Saved to: {OUT_PATH}")
