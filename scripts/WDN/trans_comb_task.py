import numpy as np
from util.task import Task 
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

class TransCombTask(Task):
    def __init__(self, species=None, masses=None, fractions=None, reactions=None, filename=None):
        self.species = species
        self.masses = masses
        self.fractions = fractions
        self.reactions = reactions
        self.filename = filename

    def _apply_to_cfg(self, cfg):
        cfg['config']['discTypes'] = [
            {'mass': self.masses[i], 'radius': np.sqrt(self.masses[i]), 'name': s}
            for i, s in enumerate(self.species)
        ]

        cfg['config']['cellMembraneType']['discTypeDistribution'] = {
            species_name: float(fraction)
            for fraction, species_name in zip(self.fractions, self.species)
        }

        cfg['config']['reactions'] = []
        for reaction in self.reactions:
            indices = reaction['indices']
            educt1 = self.species[indices[0][0]]
            educt2 = '' if len(indices[0]) == 1 else self.species[indices[0][1]]
            product1 = self.species[indices[1][0]]
            product2 = '' if len(indices[1]) == 1 else self.species[indices[1][1]]
            cfg['config']['reactions'].append({
                'educt1': educt1,
                'educt2': educt2,
                'product1': product1,
                'product2': product2,
                'probability': reaction['p']
            })

    # AI generated plot code, just for visualization of graphs
    def plot(self, output_dir: Path, large=False, dpi=100, title=True):
        graph = nx.DiGraph()
        n_species, n_reactions = len(self.species), len(self.reactions)

        # Keep a non-zero vertical span, including for one reaction.
        y_min = 0.0
        y_max = float(max(n_reactions - 1, 1))

        def distribute(count: int) -> np.ndarray:
            if count == 0:
                return np.array([])
            if count == 1:
                return np.array([(y_min + y_max) / 2])
            return np.linspace(y_min, y_max, count)

        species_y = distribute(n_species)
        reaction_y = distribute(n_reactions)

        pos = {}
        for i, species in enumerate(self.species):
            for side, x in (("L", -1), ("R", 1)):
                node = f"{side}{i}"
                graph.add_node(node, label=f"{species}: {self.masses[i]}", mass=self.masses[i])
                pos[node] = (x, species_y[i])

        colors = plt.cm.tab20(np.linspace(0, 1, min(n_reactions, 20)))
        for i, reaction in enumerate(self.reactions):
            educts, products = reaction["indices"]
            node = f"reaction_{i}"
            color = colors[i % len(colors)]
            if large:
                equation = (
                    f"{' + '.join(self.species[j] for j in educts)}\n"
                    f"→\n{' + '.join(self.species[j] for j in products)}\n"
                    f"p={reaction['p']:.2g}"
                )
            else:
                equation = (
                    f"{' + '.join(self.species[j] for j in educts)}"
                    f" → {' + '.join(self.species[j] for j in products)}"
                    f"\np={reaction['p']:.2g}"
                )

            graph.add_node(node, label=equation)
            pos[node] = (0, reaction_y[i])

            for species in educts:
                graph.add_edge(f"L{species}", node, color=color)
            for species in products:
                graph.add_edge(node, f"R{species}", color=color)

        # Remove unconnected species-side nodes.
        isolated_species_nodes = [
            node
            for node in graph.nodes
            if node.startswith(("L", "R")) and graph.degree(node) == 0
        ]
        graph.remove_nodes_from(isolated_species_nodes)

        for node in isolated_species_nodes:
            pos.pop(node, None)

        # Redistribute the remaining nodes independently on each side.
        species_nodes = []

        for side, x in (("L", -1), ("R", 1)):
            side_nodes = sorted(
                (node for node in graph.nodes if node.startswith(side)),
                key=lambda node: int(node[1:]),
            )

            for node, y in zip(side_nodes, distribute(len(side_nodes))):
                pos[node] = (x, y)

            species_nodes.extend(side_nodes)

        reaction_nodes = [
            node
            for node in graph.nodes
            if node.startswith("reaction_")
        ]

        fig, ax = plt.subplots(figsize=(11, max(4, n_reactions * 1.5)))
        factor = 2 if large else 1

        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=species_nodes,
            node_size=4000 if large else 2000,
            node_color="lightblue",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=reaction_nodes,
            node_shape="H",
            node_color="white",
            edgecolors="black",
            node_size=8000 if large else 6000,
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color=[data["color"] for _, _, data in graph.edges(data=True)],
            width=1.5,
            ax=ax,
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            labels={node: data["label"] for node, data in graph.nodes(data=True)},
            font_size=8,
            font_weight="bold",
            ax=ax,
        )

        # Must be set after NetworkX has performed its autoscaling.
        y_padding = 0.35 * factor
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        if title:
            ax.set_title(self.filename)
        ax.axis("off")
        fig.tight_layout()
        if output_dir is not None:
            fig.savefig(output_dir / Path(self.filename).with_suffix(".jpg"), dpi=dpi)
            plt.close(fig)
        else:
            plt.show()