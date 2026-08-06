import logging
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from util.paths import DATASETS_DIR, RESULTS_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SET_NAME = 'complex_transformation_set_1'
MODEL_PATH = RESULTS_DIR / f"WDN/{SET_NAME}.pt"
DATASET_PATH = DATASETS_DIR / f"WDN/{SET_NAME}.pt"


def resample_counts(df, dt=0.05):
    time_col = "ElapsedTime[s]"
    x = df[time_col].to_numpy()
    new_time = np.arange(x.min(), x.max() + dt, dt)
    count_cols = df.columns.drop(time_col)

    df_interp = pd.DataFrame({time_col: new_time})
    for col in count_cols:
        df_interp[col] = np.interp(new_time, x, df[col].to_numpy())

    N = df_interp.loc[0, count_cols].sum()
    if N > 0:
        df_interp[count_cols] /= N

    return df_interp

class ReactionData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in {
            'educt_indices',
            'product_indices',
        }:
            return len(self.x)
        return super().__inc__(key, value, *args, **kwargs)


def create_graph(species_values, reactions):
    """
    species_values: sequence such as [A, B, C], each entry a fraction in [0, 1]
    reactions: [[source_index, target_index, probability], ...]
    each graph represents a unique fraction of species
    """

    x = torch.tensor(species_values, dtype=torch.float).view(-1, 1)

    educt_indices = []
    product_indices = []
    reaction_probs = []

    for source_idx, target_idx, p in reactions:
        educt_indices.append(source_idx)
        product_indices.append(target_idx)
        reaction_probs.append([p])

    return ReactionData(
        x=x,
        educt_indices=torch.tensor(educt_indices, dtype=torch.long),
        product_indices=torch.tensor(product_indices, dtype=torch.long),
        reaction_probs=torch.tensor(reaction_probs)
    )

class ReactionDataset(Dataset):
    def __init__(self, mapping_file):
        super().__init__()
        self.samples = []
        rollout_steps = 20
        logger.info(f"Reading {mapping_file}")
        mapping = pd.read_csv(mapping_file).head(1000)

        for row in tqdm(mapping.itertuples(index=False), total=len(mapping)):
            filename = row.filename
            species = json.loads(row.species)
            reactions = json.loads(row.reactions)

            df = pd.read_csv(
                DATASETS_DIR / f"WDN/{SET_NAME}" / filename
            )
            df = resample_counts(df)

            if df.isna().any().any():
                print(filename, "resulted in nan")
                continue

            values = df[species]

            for i in range(len(df) - rollout_steps):
                graph = create_graph(
                    species_values=values.iloc[i].to_numpy(),
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


def load_dataset():
    if DATASET_PATH.exists():
        logger.info(f"Loading dataset from {DATASET_PATH}")
        dataset = torch.load(DATASET_PATH, weights_only=False)
        logger.info(f"Done loading")
        return dataset

    mapping_file = DATASETS_DIR / f"WDN/{SET_NAME}.csv"
    dataset = ReactionDataset(mapping_file)

    logger.info(f"Saving dataset to {DATASET_PATH}")
    torch.save(dataset, DATASET_PATH)
    logger.info("Done saving")
    return dataset


class ReactionGNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Message from a reactant species to its reaction node.
        # Input: [reactant_fraction, reaction_probability]
        self.reactant_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Message from reaction node to product species.
        # Input: [reaction_hidden_state, product_fraction, reaction_probability]
        self.product_mlp = nn.Sequential(
            nn.Linear(32 + 1 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        x = data.x
        reaction_educts = x[data.educt_indices]
        reaction_input = torch.cat([
            reaction_educts,
            data.reaction_probs
        ], dim=-1)
        reaction_messages = self.reactant_mlp(reaction_input)
        product_input = torch.cat([
            reaction_messages,
            x[data.product_indices],
            data.reaction_probs
        ], dim=-1)
        predicted_fraction = torch.sigmoid(self.product_mlp(product_input))
        requested_flux = predicted_fraction * data.reaction_probs * reaction_educts
        requested_outgoing = torch.zeros_like(x)
        requested_outgoing.index_add_(0, data.educt_indices, requested_flux)
        scale = torch.clamp(x / requested_outgoing.clamp_min(1e-12), max=1.0)
        flux = requested_flux * scale[data.educt_indices]
        outgoing = torch.zeros_like(x)
        outgoing.index_add_(0, data.educt_indices, flux)
        incoming = torch.zeros_like(x)
        incoming.index_add_(0, data.product_indices, flux)
        return x - outgoing + incoming


def train(device="cpu", epochs=None):
    dataset = load_dataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = ReactionGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"Number of batches: {len(loader)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(loader, leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()

            state = batch.x
            loss = 0.0

            for step in range(20):
                batch.x = state
                state = model(batch)

                prediction = state[batch.species_mask]
                target = batch.y[:, step:step + 1]

                loss = loss + loss_fn(prediction, target)

            loss /= 20
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.6g}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Saved model to {MODEL_PATH}")


def load_model(device="cpu"):
    model = ReactionGNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_trajectory(
    model,
    initial_species_values,
    reactions,
    steps,
    dt=0.05,
    device="cpu",
):
    values = list(map(float, initial_species_values))
    trajectory = [(0.0, *values)]

    with torch.no_grad():
        for step in range(1, steps + 1):
            graph = create_graph(values, reactions).to(device)
            pred = model(graph)

            values = pred[graph.species_mask].squeeze(-1).cpu().tolist()
            trajectory.append((step * dt, *values))

    return trajectory


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not MODEL_PATH.exists():
        train(device=device, epochs=5)

    model = load_model(device)

    species = ["A", "B", "C", "D", "E"]
    initial_fractions = [0.1, 0.2, 0.3, 0.2, 0.2]

    # A -> B and A -> C
    reactions = [
        [0, 1, 0.05],
        [0, 2, 0.02],
        [1, 0, 0.03],
        [4, 1, 0.01],
        [3, 1, 0.02]
    ]

    trajectory = predict_trajectory(
        model=model,
        initial_species_values=initial_fractions,
        reactions=reactions,
        steps=1200,
        dt=0.05,
        device=device,
    )
    trajectory = np.array(trajectory)

    for i, name in enumerate(species):
        plt.plot(trajectory[:, 0], trajectory[:, i+1], label=name)
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()
