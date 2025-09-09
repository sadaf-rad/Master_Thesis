import pandas as pd
import networkx as nx
import time

df = pd.read_csv(
    "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv",
    on_bad_lines='skip',
    encoding='utf-8'
)

df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])

df_2020 = df[df['timestamp'].dt.year == 2020]

G = nx.DiGraph()
G.add_edges_from(zip(df_2020['source'], df_2020['target']))
G_undirected = G.to_undirected()

nodes = list(G_undirected.nodes())
total_nodes = len(nodes)
closeness_results = {}
start_time = time.time()

print(f"Calculating closeness centrality for {total_nodes} users...")

for idx, node in enumerate(nodes):
    closeness_results[node] = nx.closeness_centrality(G_undirected, node)

    if idx % (total_nodes // 10) == 0 and idx != 0:
        percent = int((idx / total_nodes) * 100)
        elapsed = time.time() - start_time
        print(f"{percent}% done ({idx}/{total_nodes}) - elapsed: {elapsed:.1f}s")

closeness_df = pd.DataFrame.from_dict(closeness_results, orient='index', columns=['closeness_centrality'])
closeness_df.index.name = 'user_id'
closeness_df.reset_index(inplace=True)
closeness_df.to_csv("closeness_centrality_2020.csv", index=False)

print(" Done! Closeness centrality saved to 'closeness_centrality_2020.csv'")
