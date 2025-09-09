import pandas as pd
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------ Config ------------------
OUTPUT_DIR = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled"
SARAFU_CSV = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
USER_CSV = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ Step 1: Load & Filter Data ------------------
print("[1] Loading and filtering transaction data...")
df = pd.read_csv(SARAFU_CSV, on_bad_lines='skip', encoding='utf-8')
users_df = pd.read_csv(USER_CSV, low_memory=False)

df['transfer_subtype'] = df['transfer_subtype'].astype(str).str.upper().str.strip()
df['source'] = df['source'].astype(str).str.strip()
df['target'] = df['target'].astype(str).str.strip()
users_df['business_type'] = users_df['business_type'].astype(str).str.upper().str.strip()
users_df['old_POA_blockchain_address'] = users_df['old_POA_blockchain_address'].astype(str).str.strip()

# Keep only peer-to-peer (STANDARD) transactions
df = df[df['transfer_subtype'] == 'STANDARD']

# Remove system accounts
system_accounts = users_df[users_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
df = df[~df['source'].isin(system_accounts)]
df = df[~df['target'].isin(system_accounts)]

# Clean and filter by year
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df_2020 = df[df['timestamp'].dt.year == 2020].copy()

# Keep only users who actively sent transactions
active_senders = set(df_2020['source'])
df_2020 = df_2020[df_2020['source'].isin(active_senders)]

# ------------------ Step 2: Build Graph ------------------
print("[2] Creating NetworkX graph...")
G_nx = nx.from_pandas_edgelist(df_2020, 'source', 'target', create_using=nx.DiGraph())

# ------------------ Step 3: Compute Descriptive Stats ------------------
print("[3] Calculating descriptive network statistics...")

num_nodes = G_nx.number_of_nodes()
num_edges = G_nx.number_of_edges()
density = nx.density(G_nx)

# Degrees
in_degrees = dict(G_nx.in_degree())
out_degrees = dict(G_nx.out_degree())
avg_in_degree = np.mean(list(in_degrees.values()))
avg_out_degree = np.mean(list(out_degrees.values()))
max_in_degree = max(in_degrees.values())
max_out_degree = max(out_degrees.values())

# Assortativity and reciprocity
try:
    assortativity = nx.degree_pearson_correlation_coefficient(G_nx)
except:
    assortativity = np.nan

try:
    reciprocity = nx.reciprocity(G_nx)
except:
    reciprocity = np.nan

# Strongly and weakly connected components
sccs = list(nx.strongly_connected_components(G_nx))
wccs = list(nx.weakly_connected_components(G_nx))
num_scc = len(sccs)
num_wcc = len(wccs)
largest_scc_size = len(max(sccs, key=len)) if sccs else 0

# Average shortest path (on largest WCC)
largest_wcc_nodes = max(wccs, key=len) if wccs else []
G_largest_wcc = G_nx.subgraph(largest_wcc_nodes).copy()
try:
    avg_shortest_path = nx.average_shortest_path_length(G_largest_wcc)
except:
    avg_shortest_path = np.nan

# ------------------ Step 4: Print & Save ------------------
print("\n=== Descriptive Network Statistics ===")
print(f"Nodes: {num_nodes}")
print(f"Edges: {num_edges}")
print(f"Density: {density:.6f}")
print(f"Average In-Degree: {avg_in_degree:.2f}")
print(f"Average Out-Degree: {avg_out_degree:.2f}")
print(f"Max In-Degree: {max_in_degree}")
print(f"Max Out-Degree: {max_out_degree}")
print(f"Degree Assortativity: {assortativity:.4f}")
print(f"Reciprocity: {reciprocity:.4f}")
print(f"Number of SCCs: {num_scc}")
print(f"Number of WCCs: {num_wcc}")
print(f"Largest SCC Size: {largest_scc_size}")
print(f"Avg Shortest Path (Largest WCC): {avg_shortest_path:.2f}" if not np.isnan(avg_shortest_path) else "Avg Shortest Path: Not computable")

# Save to CSV
summary = {
    "nodes": num_nodes,
    "edges": num_edges,
    "density": density,
    "avg_in_degree": avg_in_degree,
    "avg_out_degree": avg_out_degree,
    "max_in_degree": max_in_degree,
    "max_out_degree": max_out_degree,
    "assortativity": assortativity,
    "reciprocity": reciprocity,
    "num_scc": num_scc,
    "num_wcc": num_wcc,
    "largest_scc_size": largest_scc_size,
    "avg_shortest_path": avg_shortest_path,
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(OUTPUT_DIR, "network_descriptives_2020.csv"), index=False)
print(f"\n[✓] Summary saved to: {os.path.join(OUTPUT_DIR, 'network_descriptives_2020.csv')}")
