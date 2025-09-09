import pandas as pd
import os

# === Paths ===
INPUT_PATH = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"
OUTPUT_DIR = "/home/s3986160/master-thesis/Retention/new definition/filtered"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "standard_users_demographics.csv")

# === Step 1: Load data ===
df = pd.read_csv(INPUT_PATH)

# === Step 2: Normalize case for text fields ===
df['business_type'] = df['business_type'].astype(str).str.lower()
df['held_roles'] = df['held_roles'].astype(str).str.lower()

# === Step 3: Filter out system/admin/vendor users ===
mask = ~df['business_type'].isin(['system']) & ~df['held_roles'].isin(['system', 'admin', 'vendor'])

# === Step 4: Remove users with no standard transactions ===
df['stxns_in'] = df['stxns_in'].fillna(0)
df['stxns_out'] = df['stxns_out'].fillna(0)
df['total_standard_txns'] = df['stxns_in'] + df['stxns_out']
mask = mask & (df['total_standard_txns'] > 0)

# === Step 5: Apply filter ===
filtered_df = df.loc[mask].copy()

# === Step 6: Select demographic columns and rename blockchain address ===
demographic_cols = ['xDAI_blockchain_address', 'gender', 'area_name', 'area_type', 'business_type', 'held_roles']
demographics = filtered_df[demographic_cols].copy()
demographics.rename(columns={'xDAI_blockchain_address': 'address'}, inplace=True)

# === Step 7: Save to CSV ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
demographics.to_csv(OUTPUT_PATH, index=False)

print(f"[✓] Saved to: {OUTPUT_PATH}")
