import networkx as nx
import matplotlib.pyplot as plt
from util.paths import REPORTS_DIR

# Create directed graph
G = nx.DiGraph()

species = ["A", "B", "C"]
reactions = ["r1", "r2"]

G.add_nodes_from(species, node_type="species")
G.add_nodes_from(reactions, node_type="reaction")

edges = [
    ("A", "r1"),
    ("B", "r1"),
    ("r1", "C"),
    ("C", "r2"),
    ("r2", "A"),
    ("r2", "B"),
]

G.add_edges_from(edges)

pos = {
    "A": (-1.5, 1),
    "B": (-1.5, -1),
    "C": (1.5, 0),
    "r1": (0, 1),
    "r2": (0, -1),
}

fig, ax = plt.subplots(figsize=(7, 5))

# Draw edges first so arrows are not hidden
nx.draw_networkx_edges(
    G, pos,
    ax=ax,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=25,
    width=1.5,
    connectionstyle="arc3,rad=0.05",
    min_source_margin=20,
    min_target_margin=25,
)

# Draw species nodes
nx.draw_networkx_nodes(
    G, pos,
    nodelist=species,
    node_color="lightblue",
    node_size=1400,
    ax=ax,
)

# Draw reaction nodes
nx.draw_networkx_nodes(
    G, pos,
    nodelist=reactions,
    node_color="lightcoral",
    node_shape="s",
    node_size=1100,
    ax=ax,
)

# Node labels
nx.draw_networkx_labels(
    G, pos,
    font_size=12,
    font_weight="bold",
    ax=ax,
)

# Legend outside the graph
species_patch = plt.Line2D(
    [0], [0],
    marker="o",
    color="w",
    markerfacecolor="lightblue",
    markersize=12,
    label="Species",
)

reaction_patch = plt.Line2D(
    [0], [0],
    marker="s",
    color="w",
    markerfacecolor="lightcoral",
    markersize=12,
    label="Reaction",
)

ax.legend(
    handles=[species_patch, reaction_patch],
    loc="upper left",
    bbox_to_anchor=(0.8, 1),
)

ax.axis("off")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "WDN/figures/reaction_graph_example.jpg", dpi=300)