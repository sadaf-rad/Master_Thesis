import pandas as pd
import numpy as np
import os
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# === File paths ===
motif_file = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/user_motif_timegaps_2020.csv"
retention_file = "/home/s3986160/master-thesis/Retention/new definition/filtered/retention_AND.csv"
out_dir = "/home/s3986160/master-thesis/Retention/new definition/filtered/labeled/compare"
os.makedirs(out_dir, exist_ok=True)

# === Load and preprocess ===
df = pd.read_csv(motif_file)
df_ret = pd.read_csv(retention_file)[["address", "retained_20"]].rename(columns={"address":"user","retained_20":"retained"})
df = df.merge(df_ret, on="user", how="inner")
df["time_days"] = df["avg_time_gap_sec"] / (3600*24)

# === Plotting KM curves ===
for motif in df["motif_type"].unique():
    subset = df[df["motif_type"] == motif]
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    for label, grp in subset.groupby("retained"):
        kmf.fit(durations=grp["time_days"],
                event_observed=np.ones(len(grp)),
                label=("Retained" if label == 1 else "Not retained"))
        kmf.plot_survival_function(ci_show=False, linestyle="-" if label==1 else "--")

    plt.title(f"Kaplan–Meier Curve: {motif.title()} Motif Completion")
    plt.xlabel("Time to Completion (days)")
    plt.ylabel("Probability Not Yet Completed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"km_{motif}_timegap_retention20.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✅ Saved: {fname}")
