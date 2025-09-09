import pandas as pd

# Step 1: Load saved degree centrality results
centrality_path = "/home/s3986160/master-thesis/Retention/new definition/DEmographics/Centralities 2020/degree_centrality_2020.csv"
df_cent = pd.read_csv(centrality_path)

# Step 2: Identify user with max degree centrality
max_row = df_cent.loc[df_cent['degree_centrality'].idxmax()]
user_id = max_row['user']
print("User with max degree centrality:")
print(max_row)

# Step 3: Load original transaction data
tx_path = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
df = pd.read_csv(tx_path, on_bad_lines='skip', encoding='utf-8')
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])

# Step 4: Filter for 2020 data
df_2020 = df[df['timestamp'].dt.year == 2020]

# Step 5: Filter transactions involving the suspicious user
df_user_txns = df_2020[(df_2020['source'] == user_id) | (df_2020['target'] == user_id)]
print(f"\nTotal transactions involving user {user_id}: {len(df_user_txns)}")
print(df_user_txns.head())

# Step 6: Check for self-loops
self_loops = df_user_txns[(df_user_txns['source'] == df_user_txns['target'])]
print(f"\nSelf-loop transactions for user {user_id}: {len(self_loops)}")
print(self_loops)

# Step 7: Check unique connection counts
unique_targets = df_user_txns[df_user_txns['source'] == user_id]['target'].nunique()
unique_sources = df_user_txns[df_user_txns['target'] == user_id]['source'].nunique()
print(f"\nUnique users this person SENT tokens to: {unique_targets}")
print(f"Unique users this person RECEIVED tokens from: {unique_sources}")
