import json
from util.paths import CONFIG_DIR, DATASETS_DIR
from util.task import Task, execute_tasks
import numpy as np
from itertools import permutations
import random
import networkx as nx
import matplotlib.pyplot as plt
import iteround
import csv

N_MAX_SPECIES = 5
N_RUNS = 5000
SET_NAME = 'complex_transformation_set_1'
random.seed(42)
np.random.seed(42)

class TransformationTask(Task):
    def __init__(self, species=None, fractions=None, reactions=None, filename=None):
        self.species = species
        self.fractions = fractions
        self.reactions = reactions
        self.filename = filename

    def _apply_to_cfg(self, cfg):
        cfg['config']['discTypes'] = [
            {'mass': 1, 'radius': 1, 'name': name}
            for name in self.species
        ]

        for fraction, species_name in zip(self.fractions, self.species):
            cfg['config']['cellMembraneType']['discTypeDistribution'][species_name] = fraction

        cfg['config']['reactions'] = []
        for reaction in self.reactions:
            cfg['config']['reactions'].append({
                'educt1': reaction['educt'],
                'educt2': '',
                'product1': reaction['product'],
                'product2': '',
                'probability': reaction['p']
            })

def generate_random_task(filename):
    n_species = np.random.randint(2, N_MAX_SPECIES + 1)
    species = [chr(ord('A') + i) for i in range(n_species)]

    alpha = np.random.uniform(0.1, 3, size=len(species))
    fractions = np.random.dirichlet(alpha=alpha)

    edge_prob = np.random.uniform(0.2, 0.8)
    possible_edges = list(permutations(species, 2))
    edges = [edge for edge in possible_edges if np.random.random() < edge_prob]
    if len(edges) == 0:
        edges = [random.choice(possible_edges)]
    reactions = []
    for edge in edges:
        reaction = dict(educt=edge[0], product=edge[1], p=np.random.uniform(1e-3, 0.1))
        reactions.append(reaction)

    return TransformationTask(species=species, fractions=fractions, reactions=reactions, filename=filename)

def plot_transformation_task(task: TransformationTask):
    G = nx.MultiDiGraph()
    G.add_nodes_from(task.species)

    for reaction in task.reactions:
        G.add_edge(
            reaction["educt"],
            reaction["product"],
            p=reaction.get("probability", reaction.get("p", 0.0)),
        )

    connectionstyle = ["arc3,rad=0.2"]

    pos = nx.circular_layout(G)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=600,
        font_weight="bold",
        connectionstyle=connectionstyle,
    )

    edge_labels = {
        (u, v, k): f"{d['p']:.2g}"
        for u, v, k, d in G.edges(keys=True, data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        connectionstyle=connectionstyle,
    )
    plt.title("Transformation Task")
    plt.show()

mapfile_path = DATASETS_DIR / f"WDN/{SET_NAME}.csv"
tasks = []
with open(mapfile_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "species", "fractions", "reactions"])
    for i in range(N_RUNS):
        filename = f"run_{i:04d}.csv"
        task = generate_random_task(filename)
        tasks.append(task)
        species = task.species
        rounded_fractions = iteround.saferound(task.fractions, 3)
        reactions = []
        for reaction in task.reactions:
            reactions.append((species.index(reaction['educt']), species.index(reaction['product']), round(reaction['p'], 3)))

        writer.writerow([
            filename,
            json.dumps(species),
            json.dumps(rounded_fractions),
            json.dumps(reactions),
        ])

with open(CONFIG_DIR / "WDN/transformation_simple.json") as f:
    cfg = json.load(f)

output_dir = DATASETS_DIR / f"WDN/{SET_NAME}"

if __name__ == "__main__":
    execute_tasks(tasks, cfg, output_dir)
