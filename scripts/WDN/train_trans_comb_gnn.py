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

SET_NAME = 'trans_comb_set_1'
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
            'educt1_indices',
            'educt2_indices',
            'product_indices',
        }:
            return len(self.x)
        return super().__inc__(key, value, *args, **kwargs)


def create_graph(species_values, reactions):
    """
    species_values: sequence such as [A, B, C], each entry a fraction in [0, 1]
    reactions:
        - unary: [educt, product, prob]
        - comb:  [educt1, educt2, product, prob]
    each graph represents a unique fraction of species
    """

    x = torch.tensor(species_values, dtype=torch.float).view(-1, 1)

    educt1_indices = []
    educt2_indices = []
    product_indices = []
    reaction_probs = []
    is_combination = []

    for reaction in reactions:
        if len(reaction) == 3:
            educt, product, prob = reaction
            educt1_indices.append(educt)
            educt2_indices.append(0) # dummy
            product_indices.append(product)
            reaction_probs.append([prob])
            is_combination.append([0])
        else:
            educt1, educt2, product, prob = reaction
            educt1_indices.append(educt1)
            educt2_indices.append(educt2)
            product_indices.append(product)
            reaction_probs.append([prob])
            is_combination.append([1])

    return ReactionData(
        x=x,
        educt1_indices=torch.tensor(educt1_indices, dtype=torch.long),
        educt2_indices=torch.tensor(educt2_indices, dtype=torch.long),
        product_indices=torch.tensor(product_indices, dtype=torch.long),
        reaction_probs=torch.tensor(reaction_probs, dtype=torch.float),
        is_combination=torch.tensor(is_combination, dtype=torch.long)
    )

class ReactionDataset(Dataset):
    def __init__(self, mapping_file):
        super().__init__()
        self.samples = []
        rollout_steps = 20
        logger.info(f"Reading {mapping_file}")
        mapping = pd.read_csv(mapping_file).head(500)

        for row in tqdm(mapping.itertuples(index=False), total=len(mapping)):
            filename = row.filename
            species = json.loads(row.species)
            reactions = json.loads(row.reactions)

            df = pd.read_csv(DATASETS_DIR / f"WDN/{SET_NAME}" /  filename)
            df = resample_counts(df)
            values = df[species]

            if not np.isfinite(values.to_numpy()).all():
                print(filename, "has non-finite entries")
                continue

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
        educt1 = x[data.educt1_indices]
        educt2 = x[data.educt2_indices]
        product = x[data.product_indices]
        input1 = torch.cat([educt1, data.reaction_probs], dim=-1)
        input2 = torch.cat([educt2, data.reaction_probs], dim=-1)
        msg1 = self.reactant_mlp(input1)
        msg2 = self.reactant_mlp(input2)
        reaction_messages = msg1 + data.is_combination * msg2
        product_input = torch.cat(
            [reaction_messages, product, data.reaction_probs],
            dim=-1
        )
        predicted_fraction = torch.sigmoid(self.product_mlp(product_input))
        educt_amount = educt1 * (
            (1.0 - data.is_combination)
            + data.is_combination * educt2
        )
        requested_flux = predicted_fraction * data.reaction_probs * educt_amount
        requested_outgoing = torch.zeros_like(x)
        requested_outgoing.index_add_(0, data.educt1_indices, requested_flux)
        requested_outgoing.index_add_(0, data.educt2_indices, requested_flux * data.is_combination)
        species_scale = torch.clamp(x / requested_outgoing.clamp_min(1e-12), max=1.0)
        scale1 = species_scale[data.educt1_indices]
        scale2 = torch.where(
            data.is_combination.bool(),
            species_scale[data.educt2_indices],
            torch.ones_like(scale1)
        )
        flux = requested_flux * torch.minimum(scale1, scale2)
        outgoing = torch.zeros_like(x)
        outgoing.index_add_(0, data.educt1_indices, flux)
        outgoing.index_add_(0, data.educt2_indices, flux * data.is_combination)
        incoming = torch.zeros_like(x)
        incoming.index_add_(0, data.product_indices, flux)
        return x - outgoing + incoming


def train(device="cpu", epochs=None):
    dataset = load_dataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

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
                if not torch.isfinite(state).all():
                    raise RuntimeError(f'Non-finite prediction at rollout step {step}')

                prediction = state
                target = batch.y[:, step:step + 1]

                loss = loss + loss_fn(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Loss is non-finite')

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

            values = pred.squeeze(-1).cpu().tolist()
            trajectory.append((step * dt, *values))

    return trajectory


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not MODEL_PATH.exists():
        train(device=device, epochs=5)

    model = load_model(device)

    species = ["A", "B", "C"]
    initial_fractions = [1, 0, 0]

    reactions = [
        [0, 1, 0.1],
        [0, 1, 2, 0.01],
        [1, 0, 0.02]
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
