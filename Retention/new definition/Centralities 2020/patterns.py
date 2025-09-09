import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# === Paths ===
TX_CSV = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
USER_CSV = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"
OUTPUT_DIR = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_graph_2020(tx_path, user_path):
    print("Loading and filtering transactions...")
    df = pd.read_csv(tx_path, on_bad_lines='skip', encoding='utf-8')
    user_df = pd.read_csv(user_path, low_memory=False)

    df['transfer_subtype'] = df['transfer_subtype'].astype(str).str.upper().str.strip()
    df['source'] = df['source'].astype(str).str.strip()
    df['target'] = df['target'].astype(str).str.strip()
    user_df['business_type'] = user_df['business_type'].astype(str).str.upper().str.strip()
    user_df['old_POA_blockchain_address'] = user_df['old_POA_blockchain_address'].astype(str).str.strip()

    df = df[df['transfer_subtype'] == 'STANDARD']
    system_accounts = user_df[user_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
    df = df[~df['source'].isin(system_accounts)]
    df = df[~df['target'].isin(system_accounts)]

    df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df[df['timestamp'].dt.year == 2020]

    active_senders = set(df['source'])
    df = df[df['source'].isin(active_senders)]

    print("Building directed graph...")
    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, tgt = row['source'], row['target']
        if G.has_edge(src, tgt):
            G[src][tgt]['weight'] += row['weight']
        else:
            G.add_edge(src, tgt, weight=row['weight'])
    return G

def label_cyclic_users(G):
    print("Labeling cyclic and acyclic users...")
    sccs = list(nx.strongly_connected_components(G))
    cyclic_nodes = set()
    for scc in sccs:
        if len(scc) > 1:
            cyclic_nodes.update(scc)
    user_df = pd.DataFrame(G.nodes(), columns=["address"])
    user_df["cyclic_status"] = user_df["address"].apply(lambda x: "cyclic" if x in cyclic_nodes else "acyclic")
    return user_df

if __name__ == "__main__":
    G_2020 = load_graph_2020(TX_CSV, USER_CSV)
    cyclic_df = label_cyclic_users(G_2020)

    out_path = f"{OUTPUT_DIR}/cyclic_users_2020.csv"
    cyclic_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
