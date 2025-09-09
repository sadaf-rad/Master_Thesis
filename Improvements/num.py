import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Use matching style
plt.style.use("seaborn-whitegrid")

# Load data
users = pd.read_csv(
    "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv",
    parse_dates=["start"]
)

# Extract and group by month
users["month"] = users["start"].dt.to_period("M").astype(str)
monthly_new_users = users.groupby("month").size().reset_index(name="new_users")

# Filter months of interest
monthly_new_users = monthly_new_users[
    (monthly_new_users["month"] >= "2020-01") & (monthly_new_users["month"] <= "2021-06")
]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(
    monthly_new_users["month"],
    monthly_new_users["new_users"],
    color="green",
    linewidth=2,
    marker="o",
    label="New Users"
)

# Shade COVID peak usage period (example: April–July 2020)
plt.axvspan("2020-04", "2020-07", color="gray", alpha=0.2)

# Add shaded legend
covid_patch = mpatches.Patch(color="gray", alpha=0.2, label="COVID-19 Peak Usage")
plt.legend(handles=[plt.gca().lines[0], covid_patch], loc="upper right")

# Aesthetics
plt.title("Number of New Registered Users per Month")
plt.xlabel("Month")
plt.ylabel("New Users")
plt.xticks(monthly_new_users["month"], rotation=90)
plt.ylim(bottom=0)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

# Save
plt.savefig("/home/s3986160/master-thesis/Improvements/09lineplot_new_users_matched.png", dpi=300)
plt.show()
