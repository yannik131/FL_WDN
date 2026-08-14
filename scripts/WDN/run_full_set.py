import json
from util.paths import CONFIG_DIR, DATASETS_DIR
from util.task import Task, execute_tasks
import numpy as np
from itertools import permutations, combinations_with_replacement, combinations
import random
import networkx as nx
import matplotlib.pyplot as plt
import iteround
import csv
from pathlib import Path
from tqdm import tqdm
from enum import IntFlag
from functools import reduce
from operator import or_
from scipy.stats import truncnorm

N_MAX_SPECIES = 10
N_RUNS = 2000
SET_NAME = 'full_set_3'
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

def plot_task(task: TransCombTask, path: Path):
    graph = nx.DiGraph()
    n_species, n_reactions = len(task.species), len(task.reactions)

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
    for i, species in enumerate(task.species):
        for side, x in (("L", -1), ("R", 1)):
            node = f"{side}{i}"
            graph.add_node(node, label=f"{species}: {task.masses[i]}", mass=task.masses[i])
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

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=species_nodes,
        node_size=2000,
        node_color="lightblue",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=reaction_nodes,
        node_shape="s",
        node_color="white",
        edgecolors="black",
        node_size=4000,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color=[data["color"] for _, _, data in graph.edges(data=True)],
        width=1.5,
        arrows=True,
        arrowsize=12,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node: data["label"] for node, data in graph.nodes(data=True)},
        font_size=8,
        ax=ax,
    )

    # Must be set after NetworkX has performed its autoscaling.
    y_padding = 0.35
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    ax.set_title(task.filename)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path / Path(task.filename).with_suffix(".jpg"), dpi=100)
    plt.close(fig)

class ReactionType(IntFlag):
    transformation = 1 << 0
    combination = 1 << 1
    decomposition = 1 << 2
    exchange = 1 << 3

def reaction_type_str(reaction_type):
    names = [x.name for x in ReactionType if x & reaction_type]
    return "|".join(names)

def sample(arr, N):
    if len(arr) <= N:
        return arr
    return random.sample(arr, N)

def sample_p(eps=1e-3):
    r = np.random.rand()

    if r < 0.70:
        mean, std = 0.1, 0.05
    elif r < 0.90:
        mean, std = 0.9, 0.05
    else:
        mean, std = 0.5, 0.20

    a = (eps - mean) / std
    b = (1 - eps - mean) / std

    return truncnorm.rvs(a, b, loc=mean, scale=std)

def generate_random_task(filename, allowed_reaction_types):
    n_species = np.random.randint(3, N_MAX_SPECIES + 1)
    species_idx = list(range(n_species))
    quantum = np.random.randint(1, 100 // (n_species - 1) + 1)
    # we'll have at least 2 identical masses and one twice as large to support all reaction types
    multipliers = [1, 1, 2]
    for _ in range(n_species - 3):
        parent_multiplier = random.choice(multipliers)
        multipliers.append(parent_multiplier + 1)
    random.shuffle(multipliers)
    masses = [quantum * multiplier for multiplier in multipliers]
    # if not all entries are equal, all reactions are possible
    if len(set(masses)) == 1:
        masses[0] = min(100, masses[0] + quantum)

    # with n species, there are ~ n^2 transformations, ~ n^3 combinations/decompositions
    # and ~ n^4 exchanges possible (very rough, small factors because of mass constraints)
    transformations = [
        [[A], [B]] for A, B in permutations(species_idx, 2)
        if masses[A] == masses[B]
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
        if masses[A] + masses[B] == masses[C] + masses[D] and (A, B) != (C, D)
    ]

    # avoid overrepresentations of reaction types with many possible edges
    lengths = [len(transformations), len(decompositions), len(combinations), len(exchanges)]
    lengths = [l for l in lengths if l > 0]
    N_min = min(lengths)
    transformations = sample(transformations, N_min) if (allowed_reaction_types & ReactionType.transformation) else []
    combinations = sample(combinations, N_min) if (allowed_reaction_types & ReactionType.combination) else []
    decompositions = sample(decompositions, N_min) if (allowed_reaction_types & ReactionType.decomposition) else []
    exchanges = sample(exchanges, N_min) if (allowed_reaction_types & ReactionType.exchange) else []

    N_reactions = len(transformations) + len(combinations) + len(decompositions) + len(exchanges)
    if N_reactions == 0:
        print(f"Fatal error: No reactions for n_species={n_species} and masses={masses} and allowed={reaction_type_str(allowed_reaction_types)}={allowed_reaction_types}")
        exit(0)

    sparsity_prob = np.random.random()
    if sparsity_prob < 0.25:
        edge_prob = min(1.0, 2.0 / N_reactions)
    elif sparsity_prob < 0.75:
        edge_prob = min(1.0, 5.0 / N_reactions)
    else:
        edge_prob = min(1.0, 10.0 / N_reactions)

    edges = [r for r in transformations if np.random.random() < edge_prob] + \
            [r for r in combinations if np.random.random() < edge_prob] + \
            [r for r in decompositions if np.random.random() < edge_prob] + \
            [r for r in exchanges if np.random.random() < edge_prob]

    if len(edges) == 0:
        edges = [random.choice(transformations + combinations + decompositions + exchanges)]

    reactions = []
    for edge in edges:
        reaction = dict(indices=edge, p=sample_p())
        reactions.append(reaction)

    used_indices = {
        i for reaction in reactions
        for side in reaction['indices']
        for i in side
    }
    used_indices = sorted(used_indices)
    mapping = {old: new for new, old in enumerate(used_indices)}

    for reaction in reactions:
        reaction['indices'] = [
            [mapping[i] for i in side]
            for side in reaction['indices']
        ]

    species = [chr(ord('A') + i) for i in range(len(used_indices))]
    masses = [masses[i] for i in used_indices]

    alpha = np.random.uniform(0.1, 3, size=len(species))
    fractions = np.random.dirichlet(alpha=alpha)

    return TransCombTask(species=species, masses=masses, fractions=fractions, reactions=reactions, filename=filename)

def generate_tasks():
    tasks = []
    reaction_types = [ReactionType.transformation, ReactionType.combination, ReactionType.decomposition, ReactionType.exchange]
    each = int(N_RUNS / 4)

    for allowed_type_count in [1, 2, 3, 4]:
        valid_combinations = list(combinations(reaction_types, allowed_type_count))
        count_for_each_combination = int(each / len(valid_combinations))
        for comb in valid_combinations:
            flag = reduce(or_, comb)
            for _ in range(count_for_each_combination):
                tasks.append(generate_random_task(f"{len(tasks):04d}.csv", flag))

    return tasks

mapfile_path = DATASETS_DIR / f"WDN/{SET_NAME}.csv"
image_dir = DATASETS_DIR / f"WDN/{SET_NAME}_figs/"
image_dir.mkdir(exist_ok=True)
tasks = []
with open(mapfile_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "species", "masses", "fractions", "reactions"])
    tasks = generate_tasks()
    for task in tqdm(tasks):
        # plot_task(task, image_dir)
        rounded_fractions = iteround.saferound(task.fractions, 3)
        reactions = []
        for reaction in task.reactions:
            reactions.append([reaction['indices'], reaction['p']])

        writer.writerow([
            task.filename,
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