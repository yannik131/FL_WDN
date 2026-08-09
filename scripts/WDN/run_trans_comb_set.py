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

N_MAX_SPECIES = 5
N_RUNS = 1000
SET_NAME = 'trans_comb_set_1'
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
        species_names = [chr(ord('A') + i) for i in self.species]
        cfg['config']['discTypes'] = [
            {'mass': self.masses[i], 'radius': np.sqrt(self.masses[i]), 'name': species_names[i]}
            for i in self.species
        ]

        for fraction, species_name in zip(self.fractions, species_names):
            cfg['config']['cellMembraneType']['discTypeDistribution'][species_name] = fraction

        cfg['config']['reactions'] = []
        for reaction in self.reactions:
            educt1 = species_names[reaction['educt1']]
            educt2 = '' if reaction['educt2'] is None else species_names[reaction['educt2']]
            product1 = species_names[reaction['product1']]
            cfg['config']['reactions'].append({
                'educt1': educt1,
                'educt2': educt2,
                'product1': product1,
                'product2': '',
                'probability': reaction['p']
            })

def generate_random_task(filename):
    n_species = np.random.randint(2, N_MAX_SPECIES + 1)
    species = list(range(n_species))
    mass_2_prob = np.random.uniform(0.2, 0.8)
    masses = [2 if np.random.random() < mass_2_prob else 1 for _ in species]

    alpha = np.random.uniform(0.1, 3, size=len(species))
    fractions = np.random.dirichlet(alpha=alpha)

    edge_prob = np.random.uniform(0.2, 0.8)
    transformation_edges = list(permutations(species, 2))
    transformation_edges = [
        (A, B) for A, B in transformation_edges
        if masses[A] == masses[B]
    ]
    combination_edges = [
        (A, B, C) for A, B in combinations_with_replacement(species, 2)
        for C in species
    ]
    combination_edges = [
        (A, B, C) for A, B, C in combination_edges
        if masses[A] + masses[B] == masses[C]
    ]
    edges = transformation_edges + combination_edges

    edges = [edge for edge in edges if np.random.random() < edge_prob]

    if len(edges) == 0:
        edges = [random.choice(transformation_edges + combination_edges)]

    reactions = []
    for edge in edges:
        educt1 = edge[0]
        educt2 = None if len(edge) == 2 else edge[1]
        product1 = edge[1] if len(edge) == 2 else edge[2]
        reaction = dict(educt1=educt1, educt2=educt2, product1=product1, p=10**np.random.uniform(-3, -1))
        reactions.append(reaction)

    return TransCombTask(species=species, masses=masses, fractions=fractions, reactions=reactions, filename=filename)

def plot_task(task: TransCombTask):
    G = nx.MultiDiGraph()
    names = {species: chr(ord("A") + species) for species in task.species}
    G.add_nodes_from(names.values())

    n_comb = sum(r["educt2"] is not None for r in task.reactions)
    comb_index = 0

    for reaction in task.reactions:
        product = names[reaction["product1"]]

        if reaction["educt2"] is None:
            G.add_edge(
                names[reaction["educt1"]],
                product,
                p=reaction["p"],
                color="gray",
            )
        else:
            color = plt.cm.hsv(comb_index / n_comb)
            comb_index += 1
            for educt in (reaction["educt1"], reaction["educt2"]):
                G.add_edge(names[educt], product, p=reaction["p"], color=color)

    pos = nx.circular_layout(G)
    connectionstyle = "arc3,rad=0.2"

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=['lightblue' if task.masses[s] == 1 else 'lightgreen' for s in task.species],
        node_size=[600 if task.masses[s] == 1 else 1200 for s in task.species],
        font_weight="bold",
        edge_color=[d["color"] for _, _, d in G.edges(data=True)],
        connectionstyle=connectionstyle,
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels={(u, v, k): f"{d['p']:.2g}" for u, v, k, d in G.edges(keys=True, data=True)},
        connectionstyle=connectionstyle,
    )

    plt.title("TransCombTask")
    plt.show()
    plt.close()

mapfile_path = DATASETS_DIR / f"WDN/{SET_NAME}.csv"
tasks = []
with open(mapfile_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "species", "masses", "fractions", "reactions"])
    for i in range(N_RUNS):
        filename = f"run_{i:04d}.csv"
        task = generate_random_task(filename)
        tasks.append(task)
        rounded_fractions = iteround.saferound(task.fractions, 3)
        reactions = []
        for reaction in task.reactions:
            if reaction['educt2'] is not None:
                reactions.append((reaction['educt1'], reaction['educt2'], reaction['product1'], round(reaction['p'], 3)))
            else:
                reactions.append((reaction['educt1'], reaction['product1'], round(reaction['p'], 3)))

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
