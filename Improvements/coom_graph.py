import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import time

# === File paths ===
TX_PATH = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
COMMUNITY_PATH = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/community_user_centralities.csv"
USER_PATH = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"
OUTPUT_PATH = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/sarafu_all_communities_graph.png"

start_time = time.time()
def log(msg):
    print(f"[{time.time() - start_time:.2f}s] {msg}")

# === Step 1: Load data ===
log("Loading transaction and user data...")
df = pd.read_csv(TX_PATH, on_bad_lines='skip', low_memory=False)
users_df = pd.read_csv(USER_PATH, low_memory=False)
community_df = pd.read_csv(COMMUNITY_PATH)

# === Step 2: Filter transactions ===
log("Filtering STANDARD transactions and removing system accounts...")
df = df[df['transfer_subtype'].str.upper().str.strip() == 'STANDARD']
system_accounts = users_df.loc[
    users_df['business_type'].str.upper().str.strip() == 'SYSTEM',
    'old_POA_blockchain_address'
].dropna().unique()
df = df[~df['source'].isin(system_accounts)]
df = df[~df['target'].isin(system_accounts)]
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df.dropna(subset=['timestamp'], inplace=True)
df_2020 = df[df['timestamp'].dt.year == 2020]

log(f"{len(df_2020)} valid transactions retained for 2020.")

# === Step 3: Build graph ===
log("Building directed graph...")
G = nx.from_pandas_edgelist(df_2020, source='source', target='target', create_using=nx.DiGraph())
log(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# === Step 4: Set community attributes ===
log("Assigning community IDs to nodes...")
community_map = community_df.set_index('user')['consensus_community_id'].to_dict()
nx.set_node_attributes(G, community_map, 'community')

nodes_with_community = [n for n in G.nodes if 'community' in G.nodes[n]]
communities = sorted(set(G.nodes[n]['community'] for n in nodes_with_community))

log(f"{len(nodes_with_community)} nodes have community assignments across {len(communities)} communities.")

# === Step 5: Color mapping ===
log("Creating color mapping...")
cmap = cm.get_cmap('tab20', len(communities)) if len(communities) <= 20 else cm.get_cmap('nipy_spectral', len(communities))
community_color_map = {com: cmap(i) for i, com in enumerate(communities)}
node_colors = [community_color_map[G.nodes[n]['community']] for n in nodes_with_community]

# === Step 6: Layout ===
log("Computing spring layout (will take a while)...")
pos = nx.spring_layout(G.subgraph(nodes_with_community), k=0.05, iterations=40, seed=42)

# === Step 7: Plot ===
log("Drawing graph...")
plt.figure(figsize=(18, 16))
nx.draw_networkx_nodes(
    G, pos,
    nodelist=nodes_with_community,
    node_color=node_colors,
    node_size=6,
    alpha=0.85
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=G.subgraph(nodes_with_community).edges(),
    alpha=0.02,
    arrows=False,
    width=0.3
)

plt.title("Full Sarafu Transaction Network (2020) ", fontsize=16)
plt.axis('off')
plt.tight_layout()

# === Step 8: Save ===
log("Saving the figure...")
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
log(f"[✅] Graph saved to: {OUTPUT_PATH}")
