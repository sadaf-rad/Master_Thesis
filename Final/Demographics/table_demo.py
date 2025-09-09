import pandas as pd
import os

# === File paths ===
base_dir = "/home/s3986160/master-thesis/Retention/new definition/filtered"
demo_path = os.path.join(base_dir, "standard_users_demographics.csv")
retention_path = os.path.join(base_dir, "retention_AND.csv")

# Load data
demographics = pd.read_csv(demo_path)
retention = pd.read_csv(retention_path)[["address", "retained_20", "not_retained_20"]]

# Merge on address
df = pd.merge(demographics, retention, on="address", how="inner")

# === Summarize by business_type ===
def summarize_business_type(df, top_n=10):
    summary = (
        df.groupby("business_type")
        .agg(Retained=('retained_20', 'sum'), Not_Retained=('not_retained_20', 'sum'))
        .reset_index()
    )
    summary["Total"] = summary["Retained"] + summary["Not_Retained"]
    summary = summary[summary["Total"] >= 10]  # filter very small groups
    summary["Retention Rate"] = (summary["Retained"] / summary["Total"]).round(4) * 100
    summary["Non-Retention Rate"] = (summary["Not_Retained"] / summary["Total"]).round(4) * 100
    summary = summary.drop(columns=["Retained", "Not_Retained", "Total"])
    return summary.sort_values("Retention Rate", ascending=False).head(top_n)

# Run summary and save
summary_business = summarize_business_type(df)
summary_business.to_csv(os.path.join(base_dir, "retention_by_business_type.csv"), index=False)
