import pandas as pd
import matplotlib.pyplot as plt
import os

# === Output directory ===
output_dir = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled"
os.makedirs(output_dir, exist_ok=True)

# === File paths ===
tx_file = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
user_file = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"

print("Loading data...")
df = pd.read_csv(tx_file, on_bad_lines='skip', encoding='utf-8')
user_df = pd.read_csv(user_file, low_memory=False)

# === Normalize columns ===
df['transfer_subtype'] = df['transfer_subtype'].astype(str).str.upper().str.strip()
df['source'] = df['source'].astype(str).str.strip()
df['target'] = df['target'].astype(str).str.strip()
user_df['business_type'] = user_df['business_type'].astype(str).str.upper().str.strip()
user_df['old_POA_blockchain_address'] = user_df['old_POA_blockchain_address'].astype(str).str.strip()

# === Filter standard + remove system accounts ===
df = df[df['transfer_subtype'] == 'STANDARD']
system_accounts = user_df[user_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
df = df[~df['source'].isin(system_accounts)]
df = df[~df['target'].isin(system_accounts)]

# === Timestamp handling ===
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])

# === Split by year ===
df_2020 = df[df['timestamp'].dt.year == 2020].copy()
df_2021 = df[df['timestamp'].dt.year == 2021].copy()

# === 2020 metrics from senders ===
tx_count_2020 = df_2020.groupby('source').size()
active_months_2020 = df_2020.groupby('source')['timestamp'].apply(lambda x: x.dt.to_period('M').nunique())
volume_2020 = df_2020.groupby('source')['weight'].sum()

valid_users_2020 = tx_count_2020.index

# === 2021 metrics (same users only) ===
df_2021_valid = df_2021[df_2021['source'].isin(valid_users_2020)]
tx_count_2021 = df_2021_valid.groupby('source').size()
active_months_2021 = df_2021_valid.groupby('source')['timestamp'].apply(lambda x: x.dt.to_period('M').nunique())
volume_2021 = df_2021_valid.groupby('source')['weight'].sum()

# === Combine + align indexes ===
metrics_2020 = pd.DataFrame({
    'tx_count': tx_count_2020,
    'active_months': active_months_2020,
    'volume': volume_2020
}).fillna(0)

metrics_2021 = pd.DataFrame({
    'tx_count': tx_count_2021,
    'active_months': active_months_2021,
    'volume': volume_2021
}).fillna(0)

metrics_2021 = metrics_2021.reindex(metrics_2020.index, fill_value=0)

# === Weighted scores ===
weights = {'tx_count': 0.5, 'active_months': 0.3, 'volume': 0.2}
score_2020 = (
    weights['tx_count'] * metrics_2020['tx_count'] +
    weights['active_months'] * metrics_2020['active_months'] +
    weights['volume'] * metrics_2020['volume']
)
score_2021 = (
    weights['tx_count'] * metrics_2021['tx_count'] +
    weights['active_months'] * metrics_2021['active_months'] +
    weights['volume'] * metrics_2021['volume']
)

# === Generate labels by threshold (AND logic only) ===
thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
retention_df = pd.DataFrame(index=metrics_2020.index)
retention_df.index.name = 'address'

summary = []
for t in thresholds:
    col = f"{int(t * 100)}"
    retained_mask = (score_2021 >= t * score_2020).astype(int)
    retention_df[f'retained_{col}'] = retained_mask
    retention_df[f'not_retained_{col}'] = 1 - retained_mask
    summary.append({
        'threshold': t,
        'retained': retained_mask.sum(),
        'not_retained': (1 - retained_mask).sum()
    })
    print(f"Threshold {col}%: Retained={retained_mask.sum()}, Not Retained={(1 - retained_mask).sum()}")

# === Save retention labels (user-level) ===
retention_csv = os.path.join(output_dir, "retention_AND.csv")
retention_df.to_csv(retention_csv)
print(f"Retention user labels saved to: {retention_csv}")

# === Save retention summary ===
summary_df = pd.DataFrame(summary)
summary_csv = os.path.join(output_dir, "retention_summary_AND.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Summary metrics saved to: {summary_csv}")

# === Plot Retained vs Not Retained (AND logic only) ===
summary_df['total'] = summary_df['retained'] + summary_df['not_retained']
summary_df['retained_pct'] = 100 * summary_df['retained'] / summary_df['total']
summary_df['not_retained_pct'] = 100 * summary_df['not_retained'] / summary_df['total']

fig, ax1 = plt.subplots(figsize=(7,5))

# Absolute counts
ax1.plot(summary_df['threshold'], summary_df['retained'], 
         marker='o', color='green', linestyle='--', label='Retained (AND)')
ax1.plot(summary_df['threshold'], summary_df['not_retained'], 
         marker='o', color='red', linestyle='--', label='Not Retained (AND)')
ax1.set_xlabel("Retention Threshold")
ax1.set_ylabel("Number of Users")

# Percentages
ax2 = ax1.twinx()
ax2.plot(summary_df['threshold'], summary_df['retained_pct'], 
         marker='o', color='green', linestyle='-', alpha=0.6)
ax2.plot(summary_df['threshold'], summary_df['not_retained_pct'], 
         marker='o', color='red', linestyle='-', alpha=0.6)
ax2.set_ylabel("Percentage of Users")

ax1.legend(loc="center right")
plt.title("Retained vs Not Retained (AND Logic)")
plt.tight_layout()

# Save figure
plot_path = os.path.join(output_dir, "andandretention_AND_plot.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"Plot saved to {plot_path}")
