import pandas as pd
import matplotlib.pyplot as plt
import os

# === Output path ===
output_dir = "/home/s3986160/master-thesis/Retention/new definition/filtered"
os.makedirs(output_dir, exist_ok=True)

# === Load data ===
print("Loading data...")
df = pd.read_csv("/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv", 
                 on_bad_lines='skip', encoding='utf-8')
user_df = pd.read_csv("/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv", 
                      low_memory=False)

# === Normalize columns ===
df['transfer_subtype'] = df['transfer_subtype'].astype(str).str.upper().str.strip()
df['source'] = df['source'].astype(str).str.strip()
df['target'] = df['target'].astype(str).str.strip()
user_df['business_type'] = user_df['business_type'].astype(str).str.upper().str.strip()
user_df['old_POA_blockchain_address'] = user_df['old_POA_blockchain_address'].astype(str).str.strip()

# === Filter: keep only standard txns ===
df = df[df['transfer_subtype'] == 'STANDARD']
print(f"Remaining after standard transaction filter: {len(df)}")

# === Filter: remove system-run users ===
system_accounts = user_df[user_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
df = df[~df['source'].isin(system_accounts)]
df = df[~df['target'].isin(system_accounts)]
print(f"Remaining after removing system accounts: {len(df)}")

# === Timestamp parsing ===
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])

# === Split by year ===
df_2020 = df[df['timestamp'].dt.year == 2020]
df_2021 = df[df['timestamp'].dt.year == 2021]

users_2020 = set(df_2020['source'].unique())
users_2021 = set(df_2021['source'].unique())
print(f"Total users active in 2020: {len(users_2020)}")
print(f"Total users active in 2021: {len(users_2021)}")

valid_users_2020 = users_2020

# === Feature extraction ===
tx_2020 = df_2020.groupby('source').size()
months_2020 = df_2020.groupby('source')['timestamp'].apply(lambda x: x.dt.to_period('M').nunique())
volume_2020 = df_2020.groupby('source')['weight'].sum()

df_2021_valid = df_2021[df_2021['source'].isin(valid_users_2020)]
tx_2021 = df_2021_valid.groupby('source').size()
months_2021 = df_2021_valid.groupby('source')['timestamp'].apply(lambda x: x.dt.to_period('M').nunique())
volume_2021 = df_2021_valid.groupby('source')['weight'].sum()

# === Combine into a DataFrame ===
df_comb = pd.DataFrame({
    'tx_2020': tx_2020,
    'months_2020': months_2020,
    'volume_2020': volume_2020,
    'tx_2021': tx_2021,
    'months_2021': months_2021,
    'volume_2021': volume_2021,
}).fillna(0)

print(f"Users considered for retention analysis: {df_comb.shape[0]} (should match users_2020)")

# === Weighting scheme ===
weights = {'tx_count': 0.5, 'active_months': 0.3, 'volume': 0.2}

df_comb['w_tx_2020'] = df_comb['tx_2020'] * weights['tx_count']
df_comb['w_tx_2021'] = df_comb['tx_2021'] * weights['tx_count']
df_comb['w_vol_2020'] = df_comb['volume_2020'] * weights['volume']
df_comb['w_vol_2021'] = df_comb['volume_2021'] * weights['volume']
df_comb['w_months_2020'] = df_comb['months_2020'] * weights['active_months']
df_comb['w_months_2021'] = df_comb['months_2021'] * weights['active_months']

# === Retention logic ===
thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
retained_counts = []
not_retained_counts = []

for t in thresholds:
    cond = (
        (df_comb['w_tx_2021'] >= t * df_comb['w_tx_2020']) |
        (df_comb['w_vol_2021'] >= t * df_comb['w_vol_2020']) |
        (df_comb['w_months_2021'] >= t * df_comb['w_months_2020'])
    )
    retained = cond.sum()
    not_retained = (~cond).sum()
    retained_counts.append(retained)
    not_retained_counts.append(not_retained)

    print(f"Threshold {int(t*100)}%: Retained={retained}, Not Retained={not_retained}")

# === Save results ===
metrics_path = os.path.join(output_dir, "user_weighted_metrics.csv")
retention_counts_path = os.path.join(output_dir, "retention_threshold_counts.csv")
plot_path = os.path.join(output_dir, "retention_or.png")

df_comb.to_csv(metrics_path, index=True)
print(f"Saved user weighted metrics to: {metrics_path}")

retention_df = pd.DataFrame({
    'threshold': thresholds,
    'retained': retained_counts,
    'not_retained': not_retained_counts
})
retention_df.to_csv(retention_counts_path, index=False)
print(f"Saved threshold retention summary to: {retention_counts_path}")

# === Plot ===
labels = [f'retained_{int(t*100)}' for t in thresholds]
x = range(len(thresholds))

plt.figure(figsize=(10, 7))
plt.bar(x, not_retained_counts, label='Not Retained', color='salmon')
plt.bar(x, retained_counts, bottom=not_retained_counts, label='Retained', color='mediumseagreen')
plt.xticks(x, labels, rotation=45)
plt.xlabel("Retention Threshold")
plt.ylabel("Number of Users")
plt.title("User Retention based on OR Logic")
plt.legend()
plt.tight_layout()

plt.savefig(plot_path, dpi=300)
print(f"Plot saved as: {plot_path}")
plt.show()
