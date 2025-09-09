import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
tx = pd.read_csv("/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv", parse_dates=["timeset"], low_memory=False)
users = pd.read_csv("/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv")

# Extract transaction month
tx["month"] = tx["timeset"].dt.to_period("M").astype(str)

# Combine all users by transaction activity
involved_users = pd.DataFrame({
    "address": pd.concat([tx["source"], tx["target"]], ignore_index=True),
    "month": pd.concat([tx["month"], tx["month"]], ignore_index=True)
})

# Merge with gender info
users = users.rename(columns={"xDAI_blockchain_address": "address"})
involved_users = involved_users.merge(users[["address", "gender"]], on="address", how="left")

# Drop NA and remove 'Other'
involved_users = involved_users.dropna(subset=["gender"])
involved_users = involved_users[~involved_users["gender"].str.lower().isin(["other"])]

# Group and normalize
grouped = involved_users.groupby(["month", "gender"])["address"].nunique().reset_index()
pivot = grouped.pivot(index="month", columns="gender", values="address").fillna(0)
pivot = pivot.div(pivot.sum(axis=1), axis=0).sort_index()

# Define custom colors
color_map = {
    "Female": "red",
    "Male": "blue",
    "Unknown": "gold"  # yellowish
}

# Plot
plt.figure(figsize=(12, 6))
for gender in pivot.columns:
    plt.plot(pivot.index, pivot[gender], marker='o', label=gender, color=color_map.get(gender, "gray"))

plt.title("Gender Proportion Over Time ")
plt.xlabel("Month")
plt.ylabel("Proportion of Users")
plt.xticks(rotation=45)
plt.ylim(0, 1.05)
plt.legend(title="Gender", loc="upper right")
plt.tight_layout()
plt.savefig("/home/s3986160/master-thesis/Improvements/gender_line_plot_colored.png", dpi=300)
plt.show()
