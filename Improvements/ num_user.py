import pandas as pd
import matplotlib.pyplot as plt

# Load users
users = pd.read_csv('/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv', parse_dates=['registration_date'])

# Extract year-month
users['month'] = users['registration_date'].dt.to_period('M').astype(str)

# Count new users per month
monthly_counts = users['month'].value_counts().sort_index()

# Plot as a line with markers
plt.figure(figsize=(10, 5))
plt.plot(monthly_counts.index, monthly_counts.values, marker='o', color='forestgreen', linewidth=2)
plt.title('Number of New Registered Users per Month')
plt.xlabel('Month')
plt.ylabel('New Users')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save it
plt.savefig('/home/s3986160/master-thesis/Improvements/new_users_line.png', dpi=300)
plt.show()
