import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

motif_type = "reciprocal"  # or "reciprocal"

# === Load data ===
motifs = pd.read_csv(
    "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_timegaps_2020.csv"
)
txns = pd.read_csv(
    "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv",
    parse_dates=['timeset'],
    on_bad_lines='skip',
    encoding='utf-8'
)

# === Clean ===
txns['source'] = txns['source'].astype(str).str.strip()
txns['target'] = txns['target'].astype(str).str.strip()
txns['transfer_subtype'] = txns['transfer_subtype'].astype(str).str.upper().str.strip()
txns = txns[txns['transfer_subtype'] == "STANDARD"]
txns = txns[txns['timeset'].dt.year == 2020]

# === Total transactions per user ===
user_tx = txns['source'].value_counts().add(txns['target'].value_counts(), fill_value=0).reset_index()
user_tx.columns = ['user', 'total_txn_count']

# === Filter by motif type and merge ===
subset = motifs[motifs['motif_type'] == motif_type]
gap_by_user = subset.groupby('user')['avg_time_gap_sec'].mean().reset_index()
df = gap_by_user.merge(user_tx, on='user', how='left').dropna()

# === Plot ===
plt.figure(figsize=(8, 6))
sc = plt.scatter(df['total_txn_count'], df['avg_time_gap_sec'],
                 c=np.log1p(df['avg_time_gap_sec']), cmap='viridis', alpha=0.7)
plt.xscale('log'); plt.yscale('log')
plt.xlabel("Total Transactions (log)")
plt.ylabel("Avg Motif Completion Time (sec, log)")
plt.title(f"{motif_type} – Completion Time vs. Total Transactions")
plt.colorbar(sc, label="log(Avg Timegap)")
plt.grid(True, which="both", ls='--')
plt.tight_layout()
plt.savefig(f"/home/s3986160/master-thesis/Plots/timegap_vs_tx_{motif_type}.png", dpi=300)
plt.show()
