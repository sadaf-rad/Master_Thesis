import pandas as pd
import numpy as np
import networkx as nx
import community.community_louvain as community
import matplotlib.pyplot as plt
import os

# === Config ===
SARAFU_CSV   = "/home/s3986160/master-thesis/Active users/sarafu_txns_20200125-20210615.csv"
RETENTION_CSV = "/home/s3986160/master-thesis/Retention/new definition/retention_AND.csv"
THRESHOLD     = 20  # default you prefer
PLOT_DIR      = "/home/s3986160/master-thesis/Plots/"

os.makedirs(PLOT_DIR, exist_ok=True)

# === Data loading & graph ===
def load_and_filter_transactions(path: str) -> pd.DataFrame:
    print("Loading transaction data...")
    df = pd.read_csv(path, on_bad_lines='skip', encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df.get('timeset'), errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df_2020 = df[df['timestamp'].dt.year == 2020].copy()
    print(f"Rows in 2020: {len(df_2020)}")
    return df_2020

def build_graph_2020(df_2020: pd.DataFrame) -> nx.DiGraph:
    print("Building 2020 graph...")
    G = nx.DiGraph()
    for _, row in df_2020.iterrows():
        src, tgt, weight = row['source'], row['target'], row['weight']
        if G.has_edge(src, tgt):
            G[src][tgt]['weight'] += weight
        else:
            G.add_edge(src, tgt, weight=weight)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def detect_communities(G: nx.DiGraph) -> pd.DataFrame:
    print("Detecting communities (Louvain on undirected projection)...")
    G_undirected = G.to_undirected()
    partition = community.best_partition(G_undirected)
    community_df = pd.DataFrame(list(partition.items()), columns=["address", "community_id"])
    total_communities = len(set(partition.values()))
    print(f"Total number of detected communities: {total_communities}")
    return community_df

def load_retention_labels(path: str, threshold: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    col = f"retained_{threshold}"
    if col not in df.columns:
        raise KeyError(f"Column {col} not found in {path}")
    df["status"] = df[col].map(lambda x: "retained" if x == 1 else "not_retained")
    return df[["address", "status"]]

# === Aggregation & analysis ===
def analyze_retention(merged_df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- RETENTION BY COMMUNITY ---")
    retention_summary = merged_df.groupby(["community_id", "status"]).size().unstack(fill_value=0)
    retention_summary["total"] = retention_summary.sum(axis=1)
    retention_summary["retention_rate"] = retention_summary.get("retained", 0) / retention_summary["total"]
    print(retention_summary.sort_values("retention_rate", ascending=False).head(10))
    return retention_summary

# === Plots ===
def plot_retention_by_community(summary_df: pd.DataFrame):
    print("Plotting retention rates by community (>5 users)...")
    df = summary_df[summary_df["total"] > 5].copy()
    if df.empty:
        print("No communities with >5 users; skipping bar plot.")
        return
    ax = df.sort_values("retention_rate", ascending=False)["retention_rate"].plot(
        kind="bar", figsize=(15, 5), color="mediumseagreen"
    )
    ax.set_title("Retention Rate by Communities with >5 Users")
    ax.set_ylabel("Retention Rate")
    ax.set_xlabel("Community ID")
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"retention_by_community_{THRESHOLD}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Plot saved as {out_path}")

def plot_size_vs_retention(summary_df: pd.DataFrame):
    """
    Scatter of community size vs retention rate with:
    - Pearson r
    - R² (variance explained)
    Only these two metrics are reported (as supervisor requested).
    """
    print("Plotting community size vs. retention rate (Pearson r and R²)...")
    df = summary_df[summary_df["total"] > 5].copy()
    if len(df) < 3:
        print("Not enough communities (>5 users) to analyze. Skipping.")
        return

    x = df["total"].astype(float).values
    y = df["retention_rate"].astype(float).values

    # Pearson correlation and R²
    pearson_r = np.corrcoef(x, y)[0, 1]
    r2 = pearson_r ** 2  # variance explained for linear fit with intercept

    # Fit line for visual reference only
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.plot(x_line, m * x_line + b, linestyle="--")

    plt.title("Community Size vs. Retention Rate")
    plt.xlabel("Community Size (number of users)")
    plt.ylabel("Retention Rate")

    # Annotate ONLY r and R²
    text = f"Pearson r = {pearson_r:.3f}\nR² = {r2:.3f}"
    plt.gca().text(
        0.02, 0.98, text,
        transform=plt.gca().transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    out_png = os.path.join(PLOT_DIR, f"size_vs_retention_{THRESHOLD}.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Plot saved as {out_png}")

    # Save numbers (r and R² only)
    out_txt = os.path.join(PLOT_DIR, f"size_vs_retention_stats_{THRESHOLD}.txt")
    with open(out_txt, "w") as f:
        f.write("Community size vs. retention rate\n")
        f.write(f"Points used (>5 users): {len(df)}\n")
        f.write(f"Pearson r: {pearson_r:.6f}\n")
        f.write(f"R^2: {r2:.6f}\n")
    print(f"Stats saved as {out_txt}")

# === Output CSV ===
def save_detailed_community_data(merged_df: pd.DataFrame, summary_df: pd.DataFrame):
    print("Saving detailed community–user data...")
    enriched = merged_df.merge(summary_df[["retention_rate", "total"]], on="community_id", how="left")
    enriched = enriched.rename(columns={"total": "community_size"})
    enriched = enriched.sort_values(["community_id", "address"])
    out_csv = f"detailed_community_users_{THRESHOLD}.csv"
    enriched.to_csv(out_csv, index=False)
    print(f"Detailed CSV saved as {out_csv}")

# === Main ===
if __name__ == "__main__":
    df_2020 = load_and_filter_transactions(SARAFU_CSV)
    G_2020 = build_graph_2020(df_2020)

    community_df = detect_communities(G_2020)
    labels_df = load_retention_labels(RETENTION_CSV, THRESHOLD)

    merged_df = pd.merge(community_df, labels_df, on="address", how="inner")
    summary_df = analyze_retention(merged_df)

    # Keep this if you also want the raw assignments
    community_df.to_csv("community_assignments_2020.csv", index=False)

    plot_retention_by_community(summary_df)
    plot_size_vs_retention(summary_df)  # now reporting ONLY r and R²
    save_detailed_community_data(merged_df, summary_df)
