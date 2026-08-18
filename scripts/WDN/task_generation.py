import numpy as np
from itertools import permutations, combinations_with_replacement, combinations
import random
from enum import IntFlag
from functools import reduce
from operator import or_
from scipy.stats import truncnorm
from WDN.trans_comb_task import TransCombTask

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

def generate_random_task(filename, allowed_reaction_types, max_species=10):
    n_species = np.random.randint(3, max_species + 1)
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

def generate_tasks(n):
    tasks = []
    reaction_types = [ReactionType.transformation, ReactionType.combination, ReactionType.decomposition, ReactionType.exchange]
    each = int(n / 4)

    for allowed_type_count in [1, 2, 3, 4]:
        valid_combinations = list(combinations(reaction_types, allowed_type_count))
        count_for_each_combination = int(each / len(valid_combinations))
        for comb in valid_combinations:
            flag = reduce(or_, comb)
            for _ in range(count_for_each_combination):
                task = generate_random_task(f"{len(tasks):04d}.csv", flag)
                tasks.append(task)

    return tasks