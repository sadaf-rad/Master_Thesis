import os
import matplotlib.pyplot as plt

# Define output directory (change if needed)
output_dir = "/home/s3986160/master-thesis/Retention/new definition/filtered/stats visuals"
os.makedirs(output_dir, exist_ok=True)

# Define centrality metrics and their mean values
metrics = {
    "Degree (Global)": {"ret": 15.85, "nonret": 7.01},
    "Degree (Community)": {"ret": 15.21, "nonret": 6.67},
    "Weighted Degree (Global)": {"ret": 11451.87, "nonret": 4537.64},
    "Weighted Degree (Community)": {"ret": 10363.67, "nonret": 4045.30},
    "Clustering (Global)": {"ret": 0.46, "nonret": 0.34},
    "Clustering (Community)": {"ret": 0.46, "nonret": 0.34},
    "Betweenness (Global)": {"ret": 0.00020, "nonret": 0.000069},
    "Betweenness (Community)": {"ret": 0.01, "nonret": 0.00}
}

# Retention rates (as %)
retention_rates = {"ret": 20, "nonret": 80}  # Adjust if using real proportions

# Generate and save each plot
for metric_name, values in metrics.items():
    fig, ax = plt.subplots()
    
    # X = centrality means, Y = retention rates
    x = [values["nonret"], values["ret"]]
    y = [retention_rates["nonret"], retention_rates["ret"]]
    labels = ['Not Retained', 'Retained']

    # Scatter plot
    ax.scatter(x, y, color=['red', 'green'])
    
    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(5,5), ha='left')

    ax.set_title(f"{metric_name} vs Retention Rate")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Retention Rate (%)")
    ax.grid(True)
    plt.tight_layout()

    # Clean filename
    safe_name = metric_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"{safe_name}.png"))
    plt.close()
