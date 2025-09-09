import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

METRIC_FILE = "/home/s3986160/master-thesis/Retention/new definition/DEmographics/Centralities 2020/clustering_coefficients_2020.csv"  # <== Change this only
LABELS_FILE = "/home/s3986160/master-thesis/Retention/new definition/retention_AND.csv"
THRESHOLD = 25



def load_labels(path, threshold):
    df = pd.read_csv(path)
    col = f"retained_{threshold}"
    df["status"] = df[col].map(lambda x: "retained" if x == 1 else "not_retained")
    return df[["address", "status"]]

def load_metric(path):
    df = pd.read_csv(path)
    if "user" in df.columns:
        df.rename(columns={"user": "address"}, inplace=True)
    return df

def compare_metric(df, metric_name, threshold):
    print(f"\nRetained users: {(df['status'] == 'retained').sum()}")
    print(f"Not retained users: {(df['status'] == 'not_retained').sum()}")

    print(f"\n--- {metric_name.upper()} DESCRIPTIVE STATS ---")
    print(df.groupby("status")[metric_name].describe())

    # Simple bar chart of means
    means = df.groupby("status")[metric_name].mean()
    means.plot(kind='bar', color=["salmon", "mediumseagreen"])
    plt.title(f"{metric_name} @ {threshold}%")
    plt.ylabel("Mean value")
    plt.tight_layout()
    plt.savefig(f"{metric_name}_{threshold}_bar.png")
    plt.close()

if __name__ == "__main__":
    metric_df = load_metric(METRIC_FILE)
    metric_name = [col for col in metric_df.columns if col != "address"][0]

    labels_df = load_labels(LABELS_FILE, THRESHOLD)
    merged_df = pd.merge(labels_df, metric_df, on="address", how="inner")

    compare_metric(merged_df, metric_name, THRESHOLD)
    print(f"\n Plot saved as {metric_name}_{THRESHOLD}_bar.png")
