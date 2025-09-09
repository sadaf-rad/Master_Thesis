import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load data ===
motifs = pd.read_csv(
    "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_timegaps_2020.csv"
)
motif_counts = pd.read_csv(
    "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_details_2020.csv"
)

# === Function to prepare data ===
def prepare_data(motif_type):
    subset_time = motifs[motifs['motif_type'] == motif_type]
    subset_count = motif_counts[motif_counts['motif_type'] == motif_type]

    gap_by_user = subset_time.groupby('user')['avg_time_gap_sec'].mean().reset_index()
    gap_by_user['avg_time_gap_days'] = gap_by_user['avg_time_gap_sec'] / (3600 * 24)  # sec → days
    count_by_user = subset_count.groupby('user')['user_count'].sum().reset_index()

    return gap_by_user.merge(count_by_user, on='user', how='inner')

# === Prepare both datasets ===
df_cycle = prepare_data("cycle_3node")
df_recip = prepare_data("reciprocal")

# === Shared color scale (based on max across both motifs) ===
all_counts = pd.concat([df_cycle['user_count'], df_recip['user_count']])
norm = plt.Normalize(vmin=np.log1p(all_counts.min()), vmax=np.log1p(all_counts.max()))

# === Plot 1: 3-node cycles ===
plt.figure(figsize=(7, 6))
sc1 = plt.scatter(df_cycle['user_count'], df_cycle['avg_time_gap_days'],
                  c=np.log1p(df_cycle['user_count']), cmap='plasma',
                  alpha=0.75, norm=norm)
plt.xscale('log')
plt.xlabel("Total Motif Count (log scale)")
plt.ylabel("Avg Completion Time (days)")
plt.title("3-Node Cycles: Completion Time vs Motif Count")
plt.colorbar(sc1, label="log(Motif Count)")
plt.grid(True, ls='--', alpha=0.6)
plt.tight_layout()
plt.savefig("/home/s3986160/master-thesis/Plots/4444timegap_vs_motifcount_cycle3_logx.png", dpi=300)
plt.close()

# === Plot 2: Reciprocal motifs ===
plt.figure(figsize=(7, 6))
sc2 = plt.scatter(df_recip['user_count'], df_recip['avg_time_gap_days'],
                  c=np.log1p(df_recip['user_count']), cmap='plasma',
                  alpha=0.75, norm=norm)
plt.xscale('log')
plt.xlabel("Total Motif Count (log scale)")
plt.ylabel("Avg Completion Time (days)")
plt.title("Reciprocal Motifs: Completion Time vs Motif Count")
plt.colorbar(sc2, label="log(Motif Count)")
plt.grid(True, ls='--', alpha=0.6)
plt.tight_layout()
plt.savefig("/home/s3986160/master-thesis/Plots/5555timegap_vs_motifcount_reciprocal_logx.png", dpi=300)
plt.close()
