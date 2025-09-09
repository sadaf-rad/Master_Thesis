import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import to_rgb
import fa2
import time

# === File paths ===
TX_PATH = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
COMMUNITY_PATH = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/community_user_centralities.csv"
OUTPUT_PATH = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/gggcccsarafu_network_gephi_style.png"

start = time.time()
def log(msg): print(f"[{time.time() - start:.1f}s] {msg}")

# === Step 1: Load Inputs ===
log("Loading files...")
tx_df = pd.read_csv(TX_PATH, low_memory=False, on_bad_lines='skip')
com_df = pd.read_csv(COMMUNITY_PATH)
filtered_users = set(com_df['user'])

# === Step 2: Filter Transactions ===
log("Filtering 2020 STANDARD transactions between filtered users...")
tx_df['timestamp'] = pd.to_datetime(tx_df['timeset'], errors='coerce')
tx_df = tx_df.dropna(subset=['timestamp'])
tx_df = tx_df[tx_df['timestamp'].dt.year == 2020]
tx_df = tx_df[tx_df['transfer_subtype'].str.upper().str.strip() == 'STANDARD']

# Keep only transactions between users in the community file
tx_df = tx_df[tx_df['source'].isin(filtered_users) & tx_df['target'].isin(filtered_users)]

log(f"{len(tx_df)} filtered transactions retained.")

# === Step 3: Build Graph ===
log("Building graph...")
G = nx.from_pandas_edgelist(tx_df, 'source', 'target', create_using=nx.Graph())

# Attach community ID as node attribute
community_map = dict(zip(com_df['user'], com_df['consensus_community_id']))
nx.set_node_attributes(G, community_map, 'community')

# Keep only nodes that have a community assigned
G = G.subgraph([n for n in G if 'community' in G.nodes[n]]).copy()

# === Step 4: Compute Layout ===
log("Computing ForceAtlas2 layout...")
forceatlas2 = fa2.ForceAtlas2(
    outboundAttractionDistribution=True,
    linLogMode=False,
    adjustSizes=False,
    edgeWeightInfluence=0.5,
    jitterTolerance=1.0,
    barnesHutOptimize=True,
    barnesHutTheta=1.2,
    scalingRatio=2,
    strongGravityMode=False,
    gravity=1.0,
    verbose=True
)
pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=500)

# === Step 5: Assign Community Colors ===
log("Assigning community colors...")
unique_communities = sorted(set(nx.get_node_attributes(G, 'community').values()))
cmap = plt.get_cmap('tab20')  # tab20 has 20 distinct colors
color_dict = {com: cmap(i % cmap.N) for i, com in enumerate(unique_communities)}
node_colors = [color_dict[G.nodes[n]['community']] for n in G.nodes]

# === Step 6: Draw Graph ===
log("Drawing final graph...")
plt.figure(figsize=(20, 18))
nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    node_size=10,
    alpha=0.9
)
nx.draw_networkx_edges(
    G, pos,
    edge_color='black',
    width=0.3,
    alpha=0.2,
    arrows=False
)

plt.axis('off')
plt.title("Sarafu Network (2020) — Community Structure", fontsize=14)
plt.tight_layout()

# === Step 7: Save ===
log("Saving to PNG...")
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
log(f"[✅] Image saved to: {OUTPUT_PATH}")
