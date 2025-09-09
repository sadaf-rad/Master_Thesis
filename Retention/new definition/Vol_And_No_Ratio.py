import pandas as pd
import matplotlib.pyplot as plt
import os

# Define output directory
output_dir = "/home/s3986160/master-thesis/Retention/new definition/filtered"
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")

# Load transaction and user data
df = pd.read_csv("/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv",
                 on_bad_lines='skip', encoding='utf-8')
user_df = pd.read_csv("/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv",
                      low_memory=False)

# Normalize and clean address and type columns
df['transfer_subtype'] = df['transfer_subtype'].astype(str).str.upper().str.strip()
df['source'] = df['source'].astype(str).str.strip()
df['target'] = df['target'].astype(str).str.strip()
user_df['business_type'] = user_df['business_type'].astype(str).str.upper().str.strip()
user_df['old_POA_blockchain_address'] = user_df['old_POA_blockchain_address'].astype(str).str.strip()

# Step 1: Keep only standard transactions
df = df[df['transfer_subtype'] == 'STANDARD']
print(f"Remaining after standard transaction filter: {len(df)}")

# Step 2: Remove transactions involving system-run accounts
system_accounts = user_df[user_df['business_type'] == 'SYSTEM']['old_POA_blockchain_address'].dropna().unique()
df = df[~df['source'].isin(system_accounts)]
df = df[~df['target'].isin(system_accounts)]
print(f"Remaining after removing system accounts: {len(df)}")

# Parse timestamps and drop invalid ones
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])

# Split into 2020 and 2021
df_2020 = df[df['timestamp'].dt.year == 2020].copy()
df_2021 = df[df['timestamp'].dt.year == 2021].copy()

# Get unique users per year
users_2020 = set(df_2020['source']).union(df_2020['target'])
users_2021 = set(df_2021['source']).union(df_2021['target'])

print(f"Number of unique users active in 2020: {len(users_2020)}")
print(f"Number of unique users active in 2021: {len(users_2021)}")

# Calculate total volume sent and received
volume_sent_2020 = df_2020.groupby('source')['weight'].sum()
volume_recv_2020 = df_2020.groupby('target')['weight'].sum()
total_volume_2020 = volume_sent_2020.add(volume_recv_2020, fill_value=0)

volume_sent_2021 = df_2021.groupby('source')['weight'].sum()
volume_recv_2021 = df_2021.groupby('target')['weight'].sum()
total_volume_2021 = volume_sent_2021.add(volume_recv_2021, fill_value=0)

# Create DataFrame of volumes
volumes = pd.DataFrame({
    'volume_2020': total_volume_2020,
    'volume_2021': total_volume_2021
}).fillna(0)

# Keep only users active in 2020
valid_users = sorted(users_2020)
volumes = volumes.reindex(valid_users, fill_value=0)

print(f"Number of users considered for retention (active in 2020): {len(volumes)}")

# Save filtered volume data
volumes_path = os.path.join(output_dir, "filtered_volume.csv")
volumes.to_csv(volumes_path)
print(f"Filtered volume data saved as '{volumes_path}'.")

# Define retention thresholds
thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
labels = [f"retained_{int(t * 100)}" for t in thresholds]
retained_counts = []
not_retained_counts = []

# Compute retention for each threshold
for t in thresholds:
    retained_users = volumes[volumes['volume_2021'] >= t * volumes['volume_2020']]
    n_retained = len(retained_users)
    n_total = len(volumes)
    retained_counts.append(n_retained)
    not_retained_counts.append(n_total - n_retained)
    
    print(f"Threshold {int(t*100)}%: Retained={n_retained}, Not Retained={n_total - n_retained}")

# Plot retention bar chart
plt.figure(figsize=(10, 7))
plt.bar(labels, not_retained_counts, label='Not Retained', color='salmon')
plt.bar(labels, retained_counts, bottom=not_retained_counts, label='Retained', color='mediumseagreen')

plt.title("User Retention Based on Total Volume")
plt.xlabel("Retention Threshold")
plt.ylabel("Number of Users")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plot_path = os.path.join(output_dir, "justvolume_total.png")
plt.savefig(plot_path, dpi=300)
print(f"Plot saved as '{plot_path}'.")

# Save summary CSV
summary_path = os.path.join(output_dir, "justvolume_total_metrics.csv")
pd.DataFrame({
    'threshold': thresholds,
    'label': labels,
    'retained_users': retained_counts,
    'not_retained_users': not_retained_counts
}).to_csv(summary_path, index=False)
print(f"Metrics saved as '{summary_path}'.")
