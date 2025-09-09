import networkx as nx
import matplotlib.pyplot as plt

# === 1. 3-node directed cycle: A → B → C → A ===
G_cycle = nx.DiGraph()
edges_cycle = [("A", "B"), ("B", "C"), ("C", "A")]
G_cycle.add_edges_from(edges_cycle)

pos_cycle = {"A": (0, 0), "B": (1, 1), "C": (2, 0)}  # 2D triangle layout

plt.figure(figsize=(5, 4))
nx.draw(G_cycle, pos_cycle, with_labels=True, node_color='lightblue', node_size=2000, font_size=14,
        edge_color='black', arrows=True, arrowsize=25)
plt.title("3-Node Directed Cycle Motif (A → B → C → A)")
plt.axis('off')
plt.tight_layout()
plt.savefig("/home/s3986160/master-thesis/Plots/motif_cycle_3node_diagram_2D.png", dpi=300)
plt.show()


# === 2. Reciprocal motif: A ↔ B ===
G_recip = nx.DiGraph()
edges_recip = [("A", "B"), ("B", "A")]
G_recip.add_edges_from(edges_recip)

pos_recip = {"A": (0, 0), "B": (2, 0)}  # 2 nodes on x-axis

plt.figure(figsize=(4, 3))
nx.draw(G_recip, pos_recip, with_labels=True, node_color='lightgreen', node_size=2000, font_size=14,
        edge_color='black', arrows=True, arrowsize=25)
plt.title("Reciprocal Motif (A ↔ B)")
plt.axis('off')
plt.tight_layout()
plt.savefig("/home/s3986160/master-thesis/Plots/motif_reciprocal_diagram_2D.png", dpi=300)
plt.show()
