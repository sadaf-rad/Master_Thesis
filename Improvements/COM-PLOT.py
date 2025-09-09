import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# Updated path
path = '/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/community_user_centralities.csv'
df = pd.read_csv(path)

# Group by community
grouped = df.groupby('consensus_community_id').agg({
    'user': 'count',
    'retained_20': 'mean'  # updated to use 20% threshold
}).reset_index()

# Rename for clarity
grouped.rename(columns={
    'user': 'community_size',
    'retained_20': 'retention_rate'
}, inplace=True)

# Linear regression
X = grouped['community_size'].values.reshape(-1, 1)
y = grouped['retention_rate'].values
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(
    grouped['community_size'],
    grouped['retention_rate'],
    s=grouped['community_size'] * 0.03,
    alpha=0.7,
    edgecolor='k'
)
plt.plot(grouped['community_size'], y_pred, color='red', linestyle='--', linewidth=2, label='Linear Regression')

# Aesthetics
plt.xlabel('Community Size')
plt.ylabel('Retention Rate')
plt.title('Retention Rate vs. Community Size')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Save to same directory
plot_path = '/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/retention_vs_community_size_linear.png'
plt.savefig(plot_path, dpi=300)
plt.close()
print(f'✅ Plot saved to: {plot_path}')
