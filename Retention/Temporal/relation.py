import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Step 1: Load your data ===
temporal_df = pd.read_csv("/home/s3986160/master-thesis/Retention/Temporal/temporal_degree_features_2020.csv")
labels_df = pd.read_csv("/home/s3986160/master-thesis/Retention/labelling/user_activity_labeled_thresholds.csv")

# === Step 2: Merge on user_id ===
df = pd.merge(temporal_df, labels_df, on="user_id")

# === Step 3: Define temporal features and label ===
temporal_features = ["degree_slope", "degree_std", "degree_mean", "degree_last", "degree_active_months", "degree_early_peak"]
target_label = "retained_15"
output_dir = "/home/s3986160/master-thesis/Retention/Temporal/"

# === Step 4: Logistic regression plots ===
for feature in temporal_features:
    plot = sns.lmplot(
        x=feature, y=target_label, data=df,
        logistic=True, ci=None,
        height=4, aspect=1.5,
        scatter_kws={"s": 20, "alpha": 0.4}
    )
    plot.fig.suptitle(f" {feature} vs {target_label}", y=1.05)
    plot.set_axis_labels(feature, "Retained (0 or 1)")

    filename = f"{output_dir}{feature}_vs_retained_15_logreg.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

# === Step 5: Correlation heatmap ===
heatmap_df = df[temporal_features + [target_label]]
correlation_matrix = heatmap_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Temporal Features vs Retention (15%)", pad=16)
plt.tight_layout()
plt.savefig(f"{output_dir}correlation_heatmap_retained_15.png")
print("✅ Saved: correlation_heatmap_retained_15.png")
plt.close()
