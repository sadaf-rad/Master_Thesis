import pandas as pd

# Load transaction data
df = pd.read_csv("/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv", parse_dates=["timeset"])

# Filter by year
df["year"] = df["timeset"].dt.year
users_2020 = set(df[df["year"] == 2020]["source"]).union(df[df["year"] == 2020]["target"])
users_2021 = set(df[df["year"] == 2021]["source"]).union(df[df["year"] == 2021]["target"])

# Users active in both years
common_users = users_2020 & users_2021

# Filter transactions involving common users
df_common = df[(df["source"].isin(common_users)) | (df["target"].isin(common_users))]

# Create unified 'user' column to track transaction per address
df_common["user"] = df_common.apply(
    lambda row: row["source"] if row["source"] in common_users else row["target"], axis=1
)

# Get first and last transaction date per user
user_activity = df_common.groupby("user")["timeset"].agg(first_txn="min", last_txn="max").reset_index()
user_activity.rename(columns={"user": "address"}, inplace=True)

# Save result
user_activity.to_csv("/home/s3986160/master-thesis/Results/users_in_both_years_txn_dates.csv", index=False)
