import pandas as pd
import os

# ------------------ File Paths ------------------
degree_path = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/raw_degree_2020.csv"
retention_path = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/retention_AND.csv"
output_path = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/normalized_degree_2020.csv"

# ------------------ Step 1: Load Data ------------------
df_degree = pd.read_csv(degree_path)  # expects columns: user, raw_degree
df_ret = pd.read_csv(retention_path)  # expects multiple retention_X, not_retained_X columns

# ------------------ Step 2: Normalize Degree ------------------
max_deg = df_degree["raw_degree"].max()
df_degree["normalized_degree"] = df_degree["raw_degree"] / max_deg

# ------------------ Step 3: Merge with Retention Labels ------------------
df = pd.merge(df_degree[["user", "normalized_degree"]], df_ret, left_on="user", right_on="address", how="left")
df = df.drop(columns=["address"])

# ------------------ Step 4: Save Output ------------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print("✅ Normalized degree file saved to:\n", output_path)
