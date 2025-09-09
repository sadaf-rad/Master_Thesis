import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("[Step 1] Load transaction data...")
tx = pd.read_csv(
    "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv",
    parse_dates=["timeset"],
    low_memory=False,
    on_bad_lines='skip'
)

# Clean and extract month
tx['month'] = tx['timeset'].dt.to_period("M").astype(str)
tx = tx[tx['month'] >= "2020-01"]  # Keep only 2020 and 2021

print("[Step 2] Aggregate monthly stats...")
monthly_stats = tx.groupby("month").agg(
    transaction_count=("timeset", "count"),
    transaction_volume=("weight", "sum")
).reset_index().sort_values("month")

print("[Step 3] Plotting dual panel...")

# Style
plt.style.use("seaborn-whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

# Volume plot
ax1.plot(monthly_stats["month"], monthly_stats["transaction_volume"], 
         color="red", linewidth=2, label="Transaction Volume")
ax1.set_title("Sum of Transactions Over Time")
ax1.set_ylabel("Total Volume (Weight)")
ax1.set_xticks(monthly_stats["month"])
ax1.set_xticklabels(monthly_stats["month"], rotation=90)
ax1.grid(True, linestyle="--", alpha=0.3)

# Count plot
ax2.plot(monthly_stats["month"], monthly_stats["transaction_count"], 
         color="blue", linewidth=2, label="Transaction Count")
ax2.set_title("Number of Transactions Over Time")
ax2.set_ylabel("Transaction Count")
ax2.set_xticks(monthly_stats["month"])
ax2.set_xticklabels(monthly_stats["month"], rotation=90)
ax2.grid(True, linestyle="--", alpha=0.3)

# Shade COVID-19 peak usage period (example: April–July 2020)
for ax in [ax1, ax2]:
    ax.axvspan("2020-04", "2020-07", color="gray", alpha=0.2)

# Add shaded legend
covid_patch = mpatches.Patch(color="gray", alpha=0.2, label="COVID-19 Peak Usage")
ax1.legend(handles=[ax1.lines[0], covid_patch], loc="upper left")
ax2.legend(handles=[ax2.lines[0], covid_patch], loc="upper left")

plt.tight_layout()
plt.savefig("/home/s3986160/master-thesis/Improvements/00transaction_growth_dual_panel_2020_2021.png", dpi=300)
plt.show()

print("✅ Saved to Improvements/00transaction_growth_dual_panel_2020_2021.png")
