# ------------------ Imports and Setup ------------------
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
import numpy as np
import os
import time
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Config ------------------
OUTPUT_DIR = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled"
SARAFU_CSV = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
USER_CSV = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"
RETENTION_CSV = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/retention_AND.csv"
THRESHOLD_COL = "retained_20"
N_RUNS = 50

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

# filter: only STANDARD and remove system accounts
df = df[df['transfer_subtype'] == 'STANDARD']
system_accounts = users_df[users_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
df = df[~df['source'].isin(system_accounts)]
df = df[~df['target'].isin(system_accounts)]

# keep 2020
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df_2020 = df[df['timestamp'].dt.year == 2020].copy()

# active senders sanity filter
active_senders = set(df_2020['source'])
df_2020 = df_2020[df_2020['source'].isin(active_senders)]

# ------------------ Step 2: Build Graph ------------------
print("[2] Creating NetworkX and igraph...")
G_nx = nx.from_pandas_edgelist(df_2020, 'source', 'target', create_using=nx.DiGraph())
edge_list = list(G_nx.edges())
users = sorted(G_nx.nodes())
print(f"[INFO] Graph has {len(users)} nodes and {len(edge_list)} edges.")

# ------------------ Step 3: Leiden Function ------------------
def run_leiden_once(run_id):
    g = ig.Graph.TupleList(edge_list, directed=True)
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    return {v["name"]: partition.membership[i] for i, v in enumerate(g.vs)}

# ------------------ Step 4: Run Leiden 50x ------------------
print("[3] Running 50 Leiden runs...")
all_runs = {user: [] for user in users}
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(run_leiden_once, i) for i in range(N_RUNS)]
    for f in tqdm(as_completed(futures), total=N_RUNS):
        result = f.result()
        for user, label in result.items():
            all_runs[user].append(label)

# Save first run
first_run_df = pd.DataFrame([(user, labels[0]) for user, labels in all_runs.items()],
                            columns=["user", "community_id"])
first_run_df.to_csv(f"{OUTPUT_DIR}/leiden_run0_assignment.csv", index=False)

# ------------------ Step 5: Mode-based Consensus ------------------
print("[4] Computing mode-based consensus labels...")
consensus_labels = {user: Counter(labels).most_common(1)[0][0] for user, labels in all_runs.items()}
df_consensus = pd.DataFrame({
    'user': list(consensus_labels.keys()),
    'consensus_community_id': list(consensus_labels.values())
})
df_consensus.to_csv(f"{OUTPUT_DIR}/leiden_consensus_assignment.csv", index=False)

# ------------------ Step 6: Community Features ------------------
print("[5] Computing community sizes...")
community_sizes = df_consensus['consensus_community_id'].value_counts().to_dict()
df_consensus['community_size'] = df_consensus['consensus_community_id'].map(community_sizes)

df_retention = pd.read_csv(RETENTION_CSV)[['address', THRESHOLD_COL]]
df_retention.rename(columns={'address': 'user'}, inplace=True)
df_combined = pd.merge(df_consensus, df_retention, on='user', how='left')
df_combined.to_csv(f"{OUTPUT_DIR}/leiden_community_features.csv", index=False)


# ------------------ Extra: Community Structure Summary ------------------
print("[6b] Community size summary stats...")

# Compute group sizes
community_sizes_series = df_combined['community_size']

# Basic stats
num_communities = df_combined['consensus_community_id'].nunique()
avg_size = community_sizes_series.mean()
min_size = community_sizes_series.min()
max_size = community_sizes_series.max()

print("Summary statistics for community structure:")
print(f"Number of communities: {num_communities}")
print(f"Average community size: {avg_size:.2f}")
print(f"Minimum community size: {min_size}")
print(f"Maximum community size: {max_size}")

# ------------------ Step 7: Descriptive Stats ------------------
print("[6] Calculating descriptive statistics...")
desc = df_combined[['community_size', THRESHOLD_COL]].describe().transpose()
desc.to_csv(f"{OUTPUT_DIR}/community_feature_descriptives.csv")

# ------------------ Step 8: Plots (+ Correlation & Variance) ------------------
print("[7] Creating plots...")
# Aggregate to community level: size and mean retention for each community
df_scatter = df_combined.groupby('consensus_community_id').agg({
    'community_size': 'first',
    THRESHOLD_COL: 'mean'
}).reset_index()

# Optional filter: keep communities with >5 users (comment out if you want all)
df_scatter_filt = df_scatter[df_scatter['community_size'] > 5].copy()
if df_scatter_filt.empty:
    df_scatter_filt = df_scatter.copy()  # fallback

x = df_scatter_filt['community_size'].astype(float).values
y = df_scatter_filt[THRESHOLD_COL].astype(float).values

# Pearson correlation and variance explained (R^2)
pearson_r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
r2 = pearson_r ** 2 if np.isfinite(pearson_r) else np.nan

# Fit line for visual reference only
if len(x) >= 2 and np.ptp(x) > 0:
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = m * x_line + b
else:
    m = b = np.nan
    x_line = y_line = None

# Scatter: Size vs. Retention with r and R^2
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_scatter_filt, x='community_size', y=THRESHOLD_COL, alpha=0.8)
if x_line is not None:
    plt.plot(x_line, y_line, linestyle='--')

plt.title("Community Size vs. Retention Rate")
plt.xlabel("Community Size")
plt.ylabel("Retention Rate")

annot = f"Pearson r = {pearson_r:.3f}\nR² = {r2:.3f}" if np.isfinite(pearson_r) else "Pearson r = n/a\nR² = n/a"
plt.gca().text(
    0.02, 0.98, annot,
    transform=plt.gca().transAxes,
    va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
)

plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
scatter_path = f"{OUTPUT_DIR}/scatter_community_size_vs_retention.png"
plt.savefig(scatter_path, dpi=300)
plt.close()

# Save stats alongside the figure
with open(f"{OUTPUT_DIR}/scatter_community_size_vs_retention_stats.txt", "w") as f:
    f.write("Community size vs. retention rate (community means)\n")
    f.write(f"Communities used (>5 users filter applied): {len(df_scatter_filt)}\n")
    f.write(f"Pearson r: {pearson_r:.6f}\n")
    f.write(f"R^2: {r2:.6f}\n")

# Bar plot: Community sizes
plt.figure(figsize=(10, 6))
df_scatter.sort_values('community_size', ascending=False).set_index('consensus_community_id')['community_size'].plot(kind='bar')
plt.title("Final Community Sizes")
plt.xlabel("Community ID")
plt.ylabel("Number of Users")
plt.tight_layout()
bar_path = f"{OUTPUT_DIR}/barplot_community_sizes.png"
plt.savefig(bar_path, dpi=300)
plt.close()

print("✅ All done! Results saved to:", OUTPUT_DIR)
print("   - Scatter PNG:", scatter_path)
print("   - Scatter stats:", f"{OUTPUT_DIR}/scatter_community_size_vs_retention_stats.txt")
print("   - Bar PNG:", bar_path)
