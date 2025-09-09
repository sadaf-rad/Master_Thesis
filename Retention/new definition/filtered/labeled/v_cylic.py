import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# === Paths ===
txn_path = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
output_dir = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/compare"
os.makedirs(output_dir, exist_ok=True)

# === Load and filter data ===
df = pd.read_csv(txn_path, on_bad_lines='skip')
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df = df[(df['timestamp'].dt.year == 2020) & df['source'].notna() & df['target'].notna()]
df['month'] = df['timestamp'].dt.to_period('M').astype(str)

# Keep only users who appeared before 2021
users_before_2021 = df.groupby('source')['timestamp'].min()
users_before_2021 = users_before_2021[users_before_2021.dt.year < 2021].index
df = df[df['source'].isin(users_before_2021) | df['target'].isin(users_before_2021)]

# === Monthly analysis ===
cyclic_counts = []
acyclic_counts = []

months = sorted(df['month'].unique())
for month in months:
    df_month = df[df['month'] == month]
    G = nx.from_pandas_edgelist(df_month, 'source', 'target', create_using=nx.DiGraph())

    # Identify strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    cyclic_users = set()
    for comp in sccs:
        if len(comp) > 1:
            cyclic_users.update(comp)
    acyclic_users = set(G.nodes()) - cyclic_users

    cyclic_counts.append(len(cyclic_users))
    acyclic_counts.append(len(acyclic_users))

# === Plotting ===
plt.figure(figsize=(12, 6))
plt.plot(months, cyclic_counts, marker='o', label='Cyclic Users', color='blue')
plt.plot(months, acyclic_counts, marker='o', label='Acyclic Users', color='green')
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Number of Users")
plt.title("Cyclic vs Acyclic Users Over Time (Filtered)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# Save
plot_path = os.path.join(output_dir, "cyclic_vs_acyclic_growth_filtered.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"✅ Plot saved to: {plot_path}")
