import pandas as pd

path = "/home/s3986160/master-thesis/Retention/labelling/centrality_betweenness_2020.csv"
df = pd.read_csv(path)

print("Head of the data:")
print(df.head())

desc = df['betweenness'].describe(percentiles=[.25, .5, .75])
print("\nDescriptive Statistics for Betweenness Centrality:")
print(desc)

mean = df['betweenness'].mean()
print(f"\nPrecise Mean: {mean:.10f}")

zero_count = (df['betweenness'] == 0).sum()
total_count = df.shape[0]
print(f"\nUsers with zero betweenness: {zero_count}/{total_count} ({(zero_count/total_count)*100:.2f}%)")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df['betweenness'], bins=100)
plt.yscale('log')
plt.title("Betweenness Centrality Distribution (Log Scale)")
plt.xlabel("Betweenness Centrality")
plt.ylabel("Frequency (log scale)")
plt.show()
