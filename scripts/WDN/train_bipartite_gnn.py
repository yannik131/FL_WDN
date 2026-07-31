import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from util.paths import DATASETS_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = RESULTS_DIR / "WDN/trans_comp_gnn.pt"
DATASET_PATH = DATASETS_DIR / "WDN/trans_comp_set_1.pt"

def resample_counts(df, dt=0.05):
    time_col = "ElapsedTime[s]"
    x = df[time_col].to_numpy()
    new_time = np.arange(x.min(), x.max() + dt, dt)
    count_cols = df.columns.drop(time_col)

    df_interp = pd.DataFrame({time_col: new_time})
    for col in count_cols:
        df_interp[col] = np.interp(new_time, x, df[col].to_numpy())

    return df_interp


def create_graph(A, B, C, N0, p_trans, p_comb):
    # [current_count, initial_total, is_reaction, prob]
    x = torch.tensor([
        [float(A), float(N0), 0, 0],
        [float(B), float(N0), 0, 0],
        [float(C), float(N0), 0, 0],
        [0, float(N0), 1, p_trans],
        [0, float(N0), 1, p_comb]
    ], dtype=torch.float)

    educt_edge_index = torch.tensor([
        [0, 0, 1],
        [3, 4, 4]
    ], dtype=torch.long)
    educt_stoich = torch.tensor([
        [1],
        [1],
        [1],
    ], dtype=torch.float)

    product_edge_index = torch.tensor([
        [3, 4],
        [1, 2]
    ], dtype=torch.float)
    product_stoich = torch.tensor([
        [1],
        [1]
    ], dtype=torch.float)

    return Data(
        x=x,
        educt_edge_index=educt_edge_index,
        educt_stoich=educt_stoich,
        product_edge_index=product_edge_index,
        product_stoich=product_stoich,
        species_mask=torch.tensor([True, True, True, False, False])
    )


class ReactionDataset(Dataset):
    def __init__(self, mapping_file):
        super().__init__()
        self.samples = []
        rollout_steps = 20
        mapping = pd.read_csv(mapping_file)

        for _, row in tqdm(mapping.iterrows(), total=len(mapping)):
            filename = row["filename"]
            N0 = float(row["N"])
            p_trans = float(row["A->B"])
            p_comb = float(row["A+B->C"])

            df = pd.read_csv(DATASETS_DIR / "WDN/trans_comp_set_1" / filename)
            df = resample_counts(df)

            if df.isna().any().any():
                print(filename, "resulted in nan")

            A = df["A"]
            B = df["B"]
            C = df["C"]

            """
            node features:
                x              y
            A:  [A(t)]      -> [A(t+dt), A(t+2*dt), ..., A(t+rollout_steps*dt)]
            B:  [B(t)]      -> [B(t+dt), B(t+2*dt), ..., B(t+rollout_steps*dt)]
            """
            for i in range(len(df) - rollout_steps):
                graph = create_graph(A.iloc[i], B.iloc[i], C.iloc[i], N0, p_trans, p_comb)
                graph.y = torch.tensor([
                    A.iloc[i + 1:i + rollout_steps + 1].to_numpy(),
                    B.iloc[i + 1:i + rollout_steps + 1].to_numpy(),
                    C.iloc[i + 1:i + rollout_steps + 1].to_numpy()
                ], dtype=torch.float)
                self.samples.append(graph)

    def len(self):
        return len(self.samples)

    def get(self, idx):
        return self.samples[idx]


def load_dataset():
    if DATASET_PATH.exists():
        logger.info(f"Loading dataset from {DATASET_PATH}")
        return torch.load(DATASET_PATH, weights_only=False)

    mapping_file = DATASETS_DIR / "WDN/trans_comp_set_1.csv"
    dataset = ReactionDataset(mapping_file)

    logger.info(f"Saving dataset to {DATASET_PATH}")
    torch.save(dataset, DATASET_PATH)
    return dataset


class ReactionGNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Message from a reactant species to a reaction node:
        # [available reactant count, initial N, reaction probability, stoichiometry]
        self.reactant_message = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Predict the fraction of the currently possible reaction extent.
        self.extent_mlp = nn.Sequential(
            nn.Linear(64 + 2, 64),  # aggregated messages, N0, p
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data, current_species_counts):
        """
        Args:
            data: Batched PyG graph.
            current_species_counts: [number_of_species_nodes_in_batch, 1].

        Returns:
            Next absolute counts for species nodes only:
            [number_of_species_nodes_in_batch, 1].
        """
        species_mask = data.species_mask
        num_nodes = data.num_nodes
        device = data.x.device

        # Insert the rolling species state into graph node features.
        current_counts = data.x[:, 0:1].clone()
        current_counts[species_mask] = current_species_counts

        react_src, react_dst = data.reactant_edge_index
        react_stoich = data.reactant_stoich

        product_src, product_dst = data.product_edge_index
        product_stoich = data.product_stoich

        N0 = data.x[:, 1:2]
        reaction_p = data.x[:, 3:4]

        # For A + B -> C, each reaction can occur at most min(A, B).
        available_per_edge = current_counts[react_src] / react_stoich

        reactant_input = torch.cat([
            available_per_edge,
            N0[react_src],
            reaction_p[react_dst],
            react_stoich,
        ], dim=-1)

        messages = self.reactant_message(reactant_input)

        # Aggregate reactant messages at each reaction node.
        aggregated = torch.zeros(
            (num_nodes, messages.size(-1)),
            device=device,
            dtype=messages.dtype,
        )
        aggregated.index_add_(0, react_dst, messages)

        degree = torch.zeros((num_nodes, 1), device=device)
        degree.index_add_(
            0,
            react_dst,
            torch.ones((react_dst.numel(), 1), device=device),
        )
        aggregated = aggregated / degree.clamp_min(1.0)

        # Limiting reagent: min(count / stoichiometry) over reactants.
        limiting_amount = torch.full(
            (num_nodes, 1),
            float("inf"),
            device=device,
        )
        limiting_amount.scatter_reduce_(
            0,
            react_dst.unsqueeze(-1),
            available_per_edge,
            reduce="amin",
            include_self=True,
        )

        reaction_nodes = ~species_mask
        limiting_amount[~reaction_nodes] = 0.0
        limiting_amount[torch.isinf(limiting_amount)] = 0.0

        extent_input = torch.cat([
            aggregated,
            N0,
            reaction_p,
        ], dim=-1)

        # One non-negative extent per reaction node.
        reaction_extent = (
            torch.sigmoid(self.extent_mlp(extent_input))
            * reaction_p
            * limiting_amount
        )
        reaction_extent[~reaction_nodes] = 0.0

        # If reactions share a reactant, do not consume more particles than exist.
        provisional_demand = torch.zeros((num_nodes, 1), device=device)
        provisional_demand.index_add_(
            0,
            react_src,
            react_stoich * reaction_extent[react_dst],
        )

        species_scale = torch.ones((num_nodes, 1), device=device)
        has_demand = provisional_demand > 0
        species_scale[has_demand] = torch.minimum(
            torch.ones_like(current_counts[has_demand]),
            current_counts[has_demand] / provisional_demand[has_demand],
        )

        reaction_scale = torch.ones((num_nodes, 1), device=device)
        reaction_scale.scatter_reduce_(
            0,
            react_dst.unsqueeze(-1),
            species_scale[react_src],
            reduce="amin",
            include_self=True,
        )
        reaction_extent = reaction_extent * reaction_scale

        # Apply stoichiometric losses and gains.
        outgoing = torch.zeros((num_nodes, 1), device=device)
        outgoing.index_add_(
            0,
            react_src,
            react_stoich * reaction_extent[react_dst],
        )

        incoming = torch.zeros((num_nodes, 1), device=device)
        incoming.index_add_(
            0,
            product_dst,
            product_stoich * reaction_extent[product_src],
        )

        next_counts = (current_counts - outgoing + incoming).clamp_min(0.0)
        return next_counts[species_mask]

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

            state = batch.x[batch.species_mask, 0:1]
            loss = 0.0

            for step in range(20):
                state = model(batch, state)
                target = batch.y[:, step:step + 1]
                loss = loss + loss_fn(state, target)

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


def predict_next_step(model, A, B, C, N0, p_a_to_b, p_a_b_to_c, device="cpu"):
    graph = create_graph(
        A=A,
        B=B,
        C=C,
        N0=N0,
        p_a_to_b=p_a_to_b,
        p_a_b_to_c=p_a_b_to_c,
    ).to(device)

    with torch.no_grad():
        state = graph.x[graph.species_mask, 0:1]
        pred = model(graph, state).squeeze(-1)

    return pred.cpu().tolist()

def predict_trajectory(
    model,
    A0,
    B0,
    C0,
    p_trans,
    p_comb,
    steps,
    dt=0.05,
    device="cpu",
):
    A = float(A0)
    B = float(B0)
    C = float(C0)
    N0 = A + B + C

    trajectory = [(0.0, A, B, C)]

    with torch.no_grad():
        for step in range(1, steps + 1):
            graph = create_graph(
                A=A,
                B=B,
                C=C,
                N0=N0,
                p_a_to_b=p_trans,
                p_a_b_to_c=p_comb,
            ).to(device)

            state = graph.x[graph.species_mask, 0:1]

            # Returns counts for species nodes [A, B, C].
            pred = model(graph, state).squeeze(-1)
            A, B, C = pred.cpu().tolist()

            trajectory.append((step * dt, A, B, C))

    return trajectory

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not MODEL_PATH.exists():
        train(device=device, epochs=5)

    model = load_model(device)

    A0 = 320
    B0 = 80
    C0 = 0

    p_trans = 0.05  # A -> B
    p_comb = 0.02   # A + B -> C

    trajectory = predict_trajectory(
        model=model,
        A0=A0,
        B0=B0,
        C0=C0,
        p_trans=p_trans,
        p_comb=p_comb,
        steps=int(60 / 0.05),
        dt=0.05,
        device=device,
    )
    trajectory = np.asarray(trajectory)

    plt.plot(trajectory[:, 0], trajectory[:, 1], label="A")
    plt.plot(trajectory[:, 0], trajectory[:, 2], label="B")
    plt.plot(trajectory[:, 0], trajectory[:, 3], label="C")

    plt.xlabel("Time [s]")
    plt.ylabel("Particle count")
    plt.legend()
    plt.tight_layout()
    plt.show()

