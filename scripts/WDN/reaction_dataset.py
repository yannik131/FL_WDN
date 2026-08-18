import torch 
from torch_geometric.data import Data, Dataset
from tqdm import tqdm 
from WDN.resample_counts import resample_counts
import json 
import pandas as pd
import numpy as np
from util.paths import DATASETS_DIR
import logging 
from sklearn.model_selection import train_test_split

logger = logging.getLogger('mylogger')

class ReactionData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in {
            'educt1_indices',
            'educt2_indices',
            'product1_indices',
            'product2_indices',
        }:
            return len(self.x)
        return super().__inc__(key, value, *args, **kwargs)


def create_graph(species_values, masses, reactions):
    """
    species_values: sequence such as [A, B, C], each entry a fraction in [0, 1]
    masses: [mass1, mass2, ...]
    reactions: [[[educt indices], [product indices]], prob]
    each graph represents a unique fraction of species
    """

    x = torch.tensor(species_values, dtype=torch.float).view(-1, 1)
    masses = torch.tensor(masses, dtype=torch.float).view(-1, 1)

    educt1_indices = []
    educt2_indices = []
    product1_indices = []
    product2_indices = []
    reaction_probs = []
    has_2_educts = []
    has_2_products = []

    for (educts, products), prob in reactions:
        educt1 = educts[0]
        educt2 = 0 if len(educts) == 1 else educts[1]
        product1 = products[0]
        product2 = 0 if len(products) == 1 else products[1]
        reaction_probs.append([prob])

        has_2_educts.append([len(educts) == 2])
        has_2_products.append([len(products) == 2])
        educt1_indices.append(educt1)
        educt2_indices.append(educt2)
        product1_indices.append(product1)
        product2_indices.append(product2)

    return ReactionData(
        x=x,
        masses=masses,
        educt1_indices=torch.tensor(educt1_indices, dtype=torch.long),
        educt2_indices=torch.tensor(educt2_indices, dtype=torch.long),
        product1_indices=torch.tensor(product1_indices, dtype=torch.long),
        product2_indices=torch.tensor(product2_indices, dtype=torch.long),
        reaction_probs=torch.tensor(reaction_probs, dtype=torch.float),
        has_2_educts=torch.tensor(has_2_educts, dtype=torch.float),
        has_2_products=torch.tensor(has_2_products, dtype=torch.float),
    )

class ReactionDataset(Dataset):
    def __init__(self, mapping, set_name):
        super().__init__()
        self.samples = []
        rollout_steps = 20

        for row in tqdm(mapping.itertuples(index=False), total=len(mapping)):
            filename = row.filename
            species = json.loads(row.species)
            masses = json.loads(row.masses)
            reactions = json.loads(row.reactions)

            df = pd.read_csv(DATASETS_DIR / f"WDN/{set_name}" /  filename)
            df = resample_counts(df)
            values = df[species]

            if not np.isfinite(values.to_numpy()).all():
                print(filename, "has non-finite entries")
                continue

            for i in range(len(df) - rollout_steps):
                graph = create_graph(
                    species_values=values.iloc[i].to_numpy(),
                    masses=masses,
                    reactions=reactions,
                )

                graph.y = torch.tensor(
                    values.iloc[i + 1:i + rollout_steps + 1].to_numpy().T,
                    dtype=torch.float,
                )

                self.samples.append(graph)

    def len(self):
        return len(self.samples)

    def get(self, idx):
        return self.samples[idx]

def load_dataset(set_name):
    train_dataset_path = DATASETS_DIR / f"WDN/{set_name}_train.pt"
    test_dataset_path = DATASETS_DIR / f"WDN/{set_name}_test.pt"

    if train_dataset_path.exists():
        logger.info(f"Loading dataset from {train_dataset_path}")
        train_dataset = torch.load(train_dataset_path, weights_only=False)
        test_dataset = torch.load(test_dataset_path, weights_only=False)
        logger.info(f"Done loading")
        return train_dataset, test_dataset

    mapping_file = DATASETS_DIR / f"WDN/{set_name}.csv"
    mapping = pd.read_csv(mapping_file)
    train_mapping, test_mapping = train_test_split(mapping, test_size=0.1, random_state=42, shuffle=True)
    train_dataset = ReactionDataset(train_mapping)
    test_dataset = ReactionDataset(test_mapping)

    logger.info(f"Saving dataset to {train_dataset_path}")
    torch.save(train_dataset, train_dataset_path)
    torch.save(test_dataset, test_dataset_path)
    logger.info("Done saving")
    return train_dataset, test_dataset