import pandas as pd
import networkx as nx

df = pd.read_csv("/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv", on_bad_lines='skip', encoding='utf-8')
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df_2020 = df[df['timestamp'].dt.year == 2020]

G = nx.DiGraph()
G.add_edges_from(zip(df_2020['source'], df_2020['target']))

G_undirected = G.to_undirected()
G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))  # ✅ Remove self-loops

kshell = nx.core_number(G_undirected)

pd.DataFrame(kshell.items(), columns=["user", "k_shell"]).to_csv("kshell_2020.csv", index=False)
