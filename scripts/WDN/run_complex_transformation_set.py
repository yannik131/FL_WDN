import json
from util.paths import CONFIG_DIR, DATASETS_DIR
from util.task import Task, execute_tasks, create_mapfile
from itertools import product
import numpy as np
from itertools import permutations
from dataclasses import dataclass
from typing import Any
import random
import networkx as nx
import matplotlib.pyplot as plt

N_MAX_SPECIES = 5
N_RUNS = 5000

@dataclass
class TransformationTask:
    species: list[str]
    fractions: np.array
    reactions: dict[str, Any] # {educt, product, probability}

def generate_random_task():
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
        reaction = dict(educt=edge[0], product=edge[1], p=np.random.uniform(0, 0.1))
        reactions.append(reaction)

    return TransformationTask(species=species, fractions=fractions, reactions=reactions)

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

def set_cfg(cfg, task: TransformationTask):
    cfg['config']['discTypes'] = [
        {'mass': 1, 'radius': 1, 'name': name}
        for name in task.species
    ]

    for fraction, species_name in zip(task.fractions, task.species):
        cfg['config']['cellMembraneType']['discTypeDistribution'][species_name] = fraction

    for reaction in task.reactions:
        cfg['config']['reactions'].append({
            'educt1': reaction['educt'],
            'educt2': '',
            'product1': reaction['product'],
            'product2': '',
            'probability': reaction['p']
        })

for _ in range(N_RUNS):
    task = generate_random_task()
    print(task)
    plot_transformation_task(task)

tasks = []
i = 0
for p, A0, B0 in product(p_vals, A0_vals, B0_vals):
    for r in range(3):  # Reduced from 5 to 3 repetitions
        N = A0 + B0
        params = dict(
            filename=f"simple_transformation_set_3_{i:04d}.csv",
            r=r,
            p=p,
            N=N,
            f_A=A0/N if N > 0 else 1,
            f_B=B0/N if N > 0 else 0
        )
        tasks.append(Task(params, mapping))
        i += 1

mapfile_path = DATASETS_DIR / "WDN/simple_transformation_set_3.csv"
create_mapfile(tasks, mapfile_path)

with open(CONFIG_DIR / "WDN/transformation_simple.json") as f:
    cfg = json.load(f)

output_dir = DATASETS_DIR / "WDN/simple_transformation_set_3"

if __name__ == "__main__":
    execute_tasks(tasks, cfg, output_dir)
