import pandas as pd
import networkx as nx
from scipy.stats import linregress

print(" Loading transaction data...")
df = pd.read_csv("sarafu_txns_20200125-20210615.csv", on_bad_lines='skip', encoding='utf-8')
df['timestamp'] = pd.to_datetime(df['timeset'], errors='coerce')
df = df.dropna(subset=['timestamp'])

print(" Filtering for 2020...")
df_2020 = df[df['timestamp'].dt.year == 2020].copy()
df_2020['month'] = df_2020['timestamp'].dt.to_period('M').astype(str)

print(" Building monthly graphs and calculating degree centrality...")
records = []

for month, df_month in df_2020.groupby('month'):
    print(f"🔹 Processing month: {month} with {len(df_month)} transactions")

    G = nx.DiGraph()
    for _, row in df_month.iterrows():
        src, tgt = row['source'], row['target']
        if G.has_edge(src, tgt):
            G[src][tgt]['weight'] += 1
        else:
            G.add_edge(src, tgt, weight=1)
    
    degree_dict = nx.degree_centrality(G)

    for user, centrality in degree_dict.items():
        records.append({
            "user_id": user,
            "month": month,
            "degree_centrality": centrality
        })

print(" Saving monthly degree centrality...")
df_centrality = pd.DataFrame(records)
df_centrality.to_csv("monthly_degree_centrality_2020.csv", index=False)


print(" Extracting temporal features...")

def extract_temporal_features(group):
    months = pd.to_datetime(group['month']).dt.month
    values = group['degree_centrality']

    # Handle edge cases: too few data points or no variance
    if len(values) < 2 or values.std() == 0:
        slope = 0
    else:
        slope, _, _, _, _ = linregress(months, values)

    return pd.Series({
        "degree_slope": slope,
        "degree_std": values.std(),
        "degree_mean": values.mean(),
        "degree_last": values.iloc[-1] if 12 in months.values else 0,
        "degree_active_months": (values > 0).sum(),
        "degree_early_peak": int(values.idxmax() in group.index[group['month'].isin(['2020-01', '2020-02', '2020-03'])])
    })

df_temporal = df_centrality.groupby("user_id").apply(extract_temporal_features).reset_index()

print(f"📦 Extracted temporal features for {len(df_temporal)} users.")
print("💾 Saving to temporal_degree_features_2020.csv...")
df_temporal.to_csv("temporal_degree_features_2020.csv", index=False)

print(" All done!")
