import json
from util.paths import CONFIG_DIR, DATASETS_DIR
from util.task import Task, execute_tasks
import numpy as np
from itertools import permutations, combinations_with_replacement
import random
import networkx as nx
import matplotlib.pyplot as plt
import iteround
import csv
from pathlib import Path
from tqdm import tqdm

N_MAX_SPECIES = 5
N_RUNS = 1000
SET_NAME = 'full_set_1'
random.seed(42)
np.random.seed(42)

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

        for fraction, species_name in zip(self.fractions, self.species):
            cfg['config']['cellMembraneType']['discTypeDistribution'][species_name] = fraction

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

def generate_random_task(filename):
    n_species = np.random.randint(2, N_MAX_SPECIES + 1)
    species = [chr(ord('A') + i) for i in range(n_species)]
    species_idx = list(range(n_species))
    mass_2_prob = np.random.uniform(0.2, 0.8)
    masses = [2 if np.random.random() < mass_2_prob else 1 for _ in species]

    alpha = np.random.uniform(0.1, 3, size=len(species))
    fractions = np.random.dirichlet(alpha=alpha)

    trans_prob = np.random.uniform(0.4, 0.8)
    comb_prob = np.random.uniform(0.3, 0.6)
    exch_prob = np.random.uniform(0.1, 0.2)
    transformations = [
        [[A], [B]] for A, B in permutations(species_idx, 2)
        if masses[A] == masses[B] and np.random.random() < trans_prob
    ]
    combinations = [
        [[A, B], [C]] for A, B in combinations_with_replacement(species_idx, 2)
        for C in species_idx
        if masses[A] + masses[B] == masses[C]
    ]
    decompositions = [reaction[::-1] for reaction in combinations]
    exchanges = [
        [[A, B], [C, D]] for A, B in combinations_with_replacement(species_idx, 2)
        for C, D in combinations_with_replacement(species_idx, 2)
        if masses[A] + masses[B] == masses[C] + masses[D] if (A, B) != (C, D)
    ]

    edges = [r for r in transformations if np.random.random() < trans_prob] + \
            [r for r in combinations if np.random.random() < comb_prob] + \
            [r for r in decompositions if np.random.random() < comb_prob] + \
            [r for r in exchanges if np.random.random() < exch_prob]

    if len(edges) == 0:
        try:
            edges = [random.choice(transformations + combinations + decompositions + exchanges)]
        except:
            print(f"Fatal error: No reactions for n_species={n_species} and masses={masses}")
            exit(0)

    reactions = []
    for edge in edges:
        reaction = dict(indices=edge, p=10**np.random.uniform(-3, -1))
        reactions.append(reaction)

    return TransCombTask(species=species, masses=masses, fractions=fractions, reactions=reactions, filename=filename)


def plot_task(task: TransCombTask, path: Path):
    graph = nx.DiGraph()
    n_species, n_reactions = len(task.species), len(task.reactions)
    species_y = np.linspace(0, max(n_reactions - 1, 1), n_species)

    pos = {}
    for i, species in enumerate(task.species):
        for side, x in (("L", -1), ("R", 1)):
            node = f"{side}{i}"
            graph.add_node(node, label=species, mass=task.masses[i])
            pos[node] = (x, species_y[i])

    for i, reaction in enumerate(task.reactions):
        educts, products = reaction["indices"]
        node = f"reaction_{i}"
        color = plt.cm.hsv(i / max(n_reactions, 1))
        equation = (
            f"{' + '.join(task.species[j] for j in educts)}"
            f" → {' + '.join(task.species[j] for j in products)}"
            f"\np={reaction['p']:.2g}"
        )

        graph.add_node(node, label=equation)
        pos[node] = (0, i)

        for species in educts:
            graph.add_edge(f"L{species}", node, color=color)
        for species in products:
            graph.add_edge(node, f"R{species}", color=color)

    fig, ax = plt.subplots(figsize=(11, max(4, n_reactions * 1.5)))
    species_nodes = [f"{side}{i}" for side in "LR" for i in range(n_species)]
    reaction_nodes = [f"reaction_{i}" for i in range(n_reactions)]

    nx.draw_networkx_nodes(
        graph, pos, nodelist=species_nodes,
        node_color=[
            "lightblue" if graph.nodes[node]["mass"] == 1 else "lightgreen"
            for node in species_nodes
        ],
        node_size=[
            1200 if graph.nodes[node]["mass"] == 1 else 2000
            for node in species_nodes
        ],
        ax=ax
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=reaction_nodes,
        node_shape="s", node_color="white",
        edgecolors="black", node_size=4000, ax=ax,
    )
    nx.draw_networkx_edges(
        graph, pos,
        edge_color=[data["color"] for _, _, data in graph.edges(data=True)],
        width=1.5, arrows=True, arrowsize=12, ax=ax,
    )
    nx.draw_networkx_labels(
        graph, pos,
        labels={node: data["label"] for node, data in graph.nodes(data=True)},
        font_size=8, ax=ax,
    )

    ax.set_title(task.filename)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path / Path(task.filename).with_suffix(".jpg"), dpi=200)
    plt.close(fig)


mapfile_path = DATASETS_DIR / f"WDN/{SET_NAME}.csv"
image_dir = DATASETS_DIR / f"WDN/{SET_NAME}_figs/"
image_dir.mkdir(exist_ok=True)
tasks = []
with open(mapfile_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "species", "masses", "fractions", "reactions"])
    for i in tqdm(list(range(N_RUNS))):
        filename = f"run_{i:04d}.csv"
        task = generate_random_task(filename)
        # plot_task(task, image_dir)
        tasks.append(task)
        rounded_fractions = iteround.saferound(task.fractions, 3)
        reactions = []
        for reaction in task.reactions:
            reactions.append([reaction['indices'], reaction['p']])

        writer.writerow([
            filename,
            json.dumps(task.species),
            json.dumps(task.masses),
            json.dumps(rounded_fractions),
            json.dumps(reactions),
        ])

with open(CONFIG_DIR / "WDN/trans_comp.json") as f:
    cfg = json.load(f)

output_dir = DATASETS_DIR / f"WDN/{SET_NAME}"

if __name__ == "__main__":
    execute_tasks(tasks, cfg, output_dir)
