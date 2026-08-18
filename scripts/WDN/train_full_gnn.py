print("Starting imports")
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
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
from time import perf_counter_ns
print("Done importing")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SET_NAME = 'full_set_3'
MODEL_PATH = RESULTS_DIR / f"WDN/{SET_NAME}.pt"
TRAIN_DATASET_PATH = DATASETS_DIR / f"WDN/{SET_NAME}_train.pt"
TEST_DATASET_PATH = DATASETS_DIR / f"WDN/{SET_NAME}_test.pt"


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
    def __init__(self, mapping):
        super().__init__()
        self.samples = []
        rollout_steps = 20

        for row in tqdm(mapping.itertuples(index=False), total=len(mapping)):
            filename = row.filename
            species = json.loads(row.species)
            masses = json.loads(row.masses)
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


def load_dataset():
    if TRAIN_DATASET_PATH.exists():
        logger.info(f"Loading dataset from {TRAIN_DATASET_PATH}")
        train_dataset = torch.load(TRAIN_DATASET_PATH, weights_only=False)
        test_dataset = torch.load(TEST_DATASET_PATH, weights_only=False)
        logger.info(f"Done loading")
        return train_dataset, test_dataset

    mapping_file = DATASETS_DIR / f"WDN/{SET_NAME}.csv"
    mapping = pd.read_csv(mapping_file)
    train_mapping, test_mapping = train_test_split(mapping, test_size=0.1, random_state=42, shuffle=True)
    train_dataset = ReactionDataset(train_mapping)
    test_dataset = ReactionDataset(test_mapping)

    logger.info(f"Saving dataset to {TRAIN_DATASET_PATH}")
    torch.save(train_dataset, TRAIN_DATASET_PATH)
    torch.save(test_dataset, TEST_DATASET_PATH)
    logger.info("Done saving")
    return train_dataset, test_dataset


class ReactionGNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Message from a reactant species to its reaction node.
        # Input: [reactant_fraction, mass, reaction_probability]
        self.reactant_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Message from reaction node to product species.
        # Input: [reaction_hidden_state, product1_fraction, product2_fraction, reaction_probability]
        self.product_mlp = nn.Sequential(
            nn.Linear(35, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        x = data.x
        educt1 = x[data.educt1_indices]
        educt2 = x[data.educt2_indices]
        educt1_mass = data.masses[data.educt1_indices]
        educt2_mass = data.masses[data.educt2_indices]
        product1 = x[data.product1_indices]
        product2 = x[data.product2_indices]
        educt_input1 = torch.cat([educt1, educt1_mass, data.reaction_probs], dim=-1)
        educt_input2 = torch.cat([educt2, educt2_mass, data.reaction_probs], dim=-1)
        msg1 = self.reactant_mlp(educt_input1)
        msg2 = self.reactant_mlp(educt_input2)
        reaction_messages = msg1 + data.has_2_educts * msg2
        product_input = torch.cat([
            reaction_messages,
            product1,
            product2 * data.has_2_products,
            data.reaction_probs
        ], dim=-1)
        predicted_fraction = torch.sigmoid(self.product_mlp(product_input))
        educt_amount = educt1 * (
            (1.0 - data.has_2_educts)
            + data.has_2_educts * educt2
        )
        requested_flux = predicted_fraction * data.reaction_probs * educt_amount
        requested_outgoing = torch.zeros_like(x)
        requested_outgoing.index_add_(0, data.educt1_indices, requested_flux)
        requested_outgoing.index_add_(0, data.educt2_indices, requested_flux * data.has_2_educts)
        species_scale = torch.clamp(x / requested_outgoing.clamp_min(1e-12), max=1.0)
        scale1 = species_scale[data.educt1_indices]
        scale2 = torch.where(
            data.has_2_educts.bool(),
            species_scale[data.educt2_indices],
            torch.ones_like(scale1)
        )
        flux = requested_flux * torch.minimum(scale1, scale2)
        outgoing = torch.zeros_like(x)
        outgoing.index_add_(0, data.educt1_indices, flux)
        outgoing.index_add_(0, data.educt2_indices, flux * data.has_2_educts)
        incoming = torch.zeros_like(x)
        incoming.index_add_(0, data.product1_indices, flux)
        incoming.index_add_(0, data.product2_indices, flux * data.has_2_products)
        return x - outgoing + incoming

def rollout_loss(model, batch, loss_fn, rollout_steps=20):
    state = batch.x
    loss = 0.0
    for step in range(rollout_steps):
        batch.x = state
        state = model(batch)
        if not torch.isfinite(state).all():
            raise RuntimeError(f'Non-finite prediction at rollout step {step}')

        target = batch.y[:, step:step + 1]

        loss = loss + loss_fn(state, target)
        if not torch.isfinite(loss):
            raise RuntimeError('Loss is non-finite')

    return loss / rollout_steps

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            loss = rollout_loss(model, batch, loss_fn)
            total_loss += loss.item()

    return total_loss / len(loader)

def train(device="cpu", epochs=None):
    train_dataset, test_dataset = load_dataset()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = ReactionGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"Number of batches: {len(train_loader)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()

            loss = rollout_loss(model, batch, loss_fn)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.6g}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Saved model to {MODEL_PATH}")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss:.6g}")

def load_model(device="cpu"):
    model = ReactionGNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_trajectory(
    model,
    initial_species_values,
    masses,
    reactions,
    steps,
    dt=0.05,
    device="cpu",
):
    values = list(map(float, initial_species_values))
    trajectory = [(0.0, *values)]

    print("Starting prediction")
    start = perf_counter_ns()
    with torch.no_grad():
        for step in range(1, steps + 1):
            graph = create_graph(values, masses, reactions).to(device)
            pred = model(graph)

            values = pred.squeeze(-1).cpu().tolist()
            trajectory.append((step * dt, *values))
    dt = (perf_counter_ns() - start) / 1e9
    print(f"Prediction took {dt:.3g}s")

    return trajectory

def water_example(model, device):
    species = ["H+", "H20", "Ca²⁺", "CO2g", "CO2aq", "CO₃²⁻", "HCO3-", "H2CO3", "CaCO₃ (s)"]
    initial_fractions = [0, 0.8, 0.1, 0.1, 0, 0, 0, 0, 0]

    masses = [1, 18, 40,   44,  44,   60,   61,   62,   100]
    #         H+ H20 Ca2+  CO2g CO2aq CO32- HCO3- H2CO3 CaCO3
    #         0  1   2     3    4     5     6     7     8
    reactions = [
        [[[3], [4]], 0.05],
        [[[4], [3]], 0.01],
        [[[4, 1], [7]], 0.02],
        [[[7], [4, 1]], 0.2],
        [[[7], [0, 6]], 0.865],
        [[[0, 6], [7]], 0.12],
        [[[6], [0, 5]], 0.03],
        [[[0, 5], [6]], 0.3],
        [[[8], [2, 5]], 0.002],
        [[[2, 5], [8]], 0.015],
    ]

    trajectory = predict_trajectory(
        model=model,
        initial_species_values=initial_fractions,
        masses=masses,
        reactions=reactions,
        steps=int(1000 / 0.05),
        dt=0.05,
        device=device,
    )
    trajectory = np.array(trajectory)
    fig, ax = plt.subplots()

    colors = ['blue', 'red', 'green']
    df = pd.read_csv(RESULTS_DIR / "WDN/water_averaged_df.csv")
    df = resample_counts(df)

    for color, name in zip(colors, ['Ca²⁺', 'CO₃²⁻', 'CaCO₃ (s)']):
        i = species.index(name)
        ax.plot(trajectory[:, 0], trajectory[:, i+1], linestyle="--", color=color)
        ax.plot(trajectory[:, 0], df[name].to_numpy(), color=color, label=name)

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color='black', linestyle='-', label='averaged'),
        Line2D([0], [0], color='black', linestyle='--', label='predicted')
    ]

    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend(handles=handles)
    plt.tight_layout()
    plt.show()

def lv_example(model, devie):
    species = ["Prey", "Predator", "Resource"]
    initial_fractions = [0, 0, 1]

    masses = [30,  30,      30]
    #         Prey Predator Resource
    #         0    1        2
    reactions = [
        [[[0, 1], [1, 1]], 0.5],
        [[[0, 2], [0, 0]], 0.02],
        [[[0], [2]], 0.05],
        [[[1], [2]], 0.9],
        [[[2], [0]], 0.01],
        [[[2], [1]], 0.01]
    ]

    trajectory = predict_trajectory(
        model=model,
        initial_species_values=initial_fractions,
        masses=masses,
        reactions=reactions,
        steps=int(120 / 0.05),
        dt=0.05,
        device=device,
    )
    trajectory = np.array(trajectory)

    for name in species:
        i = species.index(name)
        plt.plot(trajectory[:, 0], trajectory[:, i+1], label=name)

    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = "cpu"
    print(f"Using device: {device}")

    if not MODEL_PATH.exists():
        train(device=device, epochs=5)
        exit(0)

    model = load_model(device)
    lv_example(model, device)
    
