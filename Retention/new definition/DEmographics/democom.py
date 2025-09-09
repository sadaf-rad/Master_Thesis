import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DETAILS_CSV = "/home/s3986160/master-thesis/detailed_community_users_25.csv"
USERS_CSV = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"

DEMOGRAPHIC_COLUMNS = ["gender", "area_name", "business_type"]

def load_and_merge_data(details_path, users_path):
    print("Loading and merging data...")
    details_df = pd.read_csv(details_path)
    users_df = pd.read_csv(users_path, low_memory=False)

    if "xDAI_blockchain_address" in users_df.columns:
        users_df.rename(columns={"xDAI_blockchain_address": "address"}, inplace=True)
    elif "old_POA_blockchain_address" in users_df.columns:
        users_df.rename(columns={"old_POA_blockchain_address": "address"}, inplace=True)
    else:
        raise KeyError("No recognizable blockchain address column found.")

    merged = pd.merge(details_df, users_df, on="address", how="left")

    for col in DEMOGRAPHIC_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).str.lower().str.strip()

    return merged

def plot_demographic_heatmap(df, demographic_col, min_size=10):
    print(f"Plotting heatmap for: {demographic_col}")
    counts = df.groupby(["community_id", demographic_col]).size().unstack(fill_value=0)
    community_sizes = df.groupby("community_id").size()
    counts = counts.loc[community_sizes[community_sizes >= min_size].index]
    proportions = counts.div(counts.sum(axis=1), axis=0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(proportions, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title(f"Proportion of {demographic_col} per Community")
    plt.tight_layout()
    plt.savefig(f"heatmap_{demographic_col}.png")
    plt.close()
    print(f" Saved heatmap_{demographic_col}.png")

def plot_all_demo_values_vs_retention(df, demographic_col):
    print(f"Plotting scatter plots for all values in: {demographic_col}")
    demo_counts = df.groupby(["community_id", demographic_col]).size().unstack(fill_value=0)
    retention = df.groupby("community_id")["retention_rate"].mean()
    total = demo_counts.sum(axis=1)

    for value in demo_counts.columns:
        target_prop = demo_counts[value] / total

        plt.figure(figsize=(8, 6))
        plt.scatter(target_prop, retention, alpha=0.7)
        plt.title(f"% {value} in {demographic_col} vs. Retention Rate")
        plt.xlabel(f"% {value}")
        plt.ylabel("Retention Rate")
        plt.tight_layout()
        safe_value = str(value).replace(" ", "_").replace("/", "_")
        plt.savefig(f"scatter_{demographic_col}_{safe_value}_vs_retention.png")
        plt.close()
        print(f" Saved scatter_{demographic_col}_{safe_value}_vs_retention.png")

def plot_top_community_bars(df, demographic_col, top_n=10):
    print(f"Plotting grouped bar chart for top {top_n} communities: {demographic_col}")
    top_communities = df.groupby("community_id").size().sort_values(ascending=False).head(top_n).index
    subset = df[df["community_id"].isin(top_communities)]

    counts = subset.groupby(["community_id", demographic_col]).size().unstack(fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)
    retention = subset.groupby("community_id")["retention_rate"].mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    proportions.plot(kind='bar', stacked=True, ax=ax, colormap="tab20c")
    retention.plot(style='o--', color="black", secondary_y=True, ax=ax)

    ax.set_title(f"Demographics + Retention for Top {top_n} Communities ({demographic_col})")
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Proportion of Demographic")
    ax.right_ax.set_ylabel("Retention Rate")
    plt.tight_layout()
    plt.savefig(f"grouped_bar_top_communities_{demographic_col}.png")
    plt.close()
    print(f" Saved grouped_bar_top_communities_{demographic_col}.png")

if __name__ == "__main__":
    merged_df = load_and_merge_data(DETAILS_CSV, USERS_CSV)

    for col in DEMOGRAPHIC_COLUMNS:
        plot_demographic_heatmap(merged_df, demographic_col=col)
        plot_all_demo_values_vs_retention(merged_df, demographic_col=col)
        plot_top_community_bars(merged_df, demographic_col=col)
