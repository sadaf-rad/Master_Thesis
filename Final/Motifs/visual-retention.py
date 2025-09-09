import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === File paths ===
motif_timegap_path = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_timegaps_2020.csv"
retention_path     = "/home/s3986160/master-thesis/Retention/new definition/retention_AND.csv"
retention_column   = "retained_20"    # or retained_25

# === Load data ===
motifs    = pd.read_csv(motif_timegap_path)
retention = pd.read_csv(retention_path)[['address', retention_column]]
retention.columns = ['user', 'retained']

# === Clean IDs ===
motifs['user']    = motifs['user'].astype(str).str.lower().str.strip()
retention['user'] = retention['user'].astype(str).str.lower().str.strip()

# === Merge with retention ===
df = motifs.merge(retention, on='user', how='inner')
df['avg_time_gap_days'] = df['avg_time_gap_sec'] / (3600 * 24)

# === Plot all in one ===
plt.figure(figsize=(10, 6))

# Custom colors
colors = {
    ("cycle_3node", 0): "red",
    ("cycle_3node", 1): "green",
    ("reciprocal", 0): "orange",
    ("reciprocal", 1): "blue"
}

labels = {
    ("cycle_3node", 0): "3-Node Cycle – Not Retained",
    ("cycle_3node", 1): "3-Node Cycle – Retained",
    ("reciprocal", 0): "Reciprocal – Not Retained",
    ("reciprocal", 1): "Reciprocal – Retained"
}

for motif_type in ["cycle_3node", "reciprocal"]:
    for retained_val in [0, 1]:
        subset = df[(df['motif_type'] == motif_type) & (df['retained'] == retained_val)]
        if not subset.empty:
            sns.kdeplot(
                subset['avg_time_gap_days'],
                label=labels[(motif_type, retained_val)],
                color=colors[(motif_type, retained_val)],
                lw=2,
                fill=False
            )

plt.xlabel("Avg Motif Completion Time (days)")
plt.ylabel("Density")
plt.title("Motif Completion Time by Retention (3-Node vs Reciprocal)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# === Save ===
outpath = "/home/s3986160/master-thesis/Plots/1hist_sameaxis_timegap_vs_retention_allmotifs.png"
plt.savefig(outpath, dpi=300)
plt.show()
print(f"Saved: {outpath}")
