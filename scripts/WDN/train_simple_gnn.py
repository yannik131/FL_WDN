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

"""
In this simple first approach we model a single reaction A -> B with a simple graph:
A -> B
where the reaction probability is an edge attribute of the edge (A, B). 
"""


"""
task:
compare prediction with analytical solution and averaged 100 runs and compare intersection time with 100 time histogram
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = RESULTS_DIR / "WDN/simple_gnn_transformation_3.pt"
DATASET_PATH = DATASETS_DIR / "WDN/simple_transformation_data_flux_3.pt"


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


def create_graph(A, B, p):
    x = torch.tensor([
        [float(A)], # source nodes
        [float(B)], # target nodes
    ], dtype=torch.float)

    edge_index = torch.tensor([
        [0], # all source node indices
        [1], # corresponding target node indices
    ], dtype=torch.long)

    edge_attr = torch.tensor([
        [float(p)] # attribute for first edge
    ], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )


class ReactionDataset(Dataset):
    def __init__(self, mapping_file):
        super().__init__()
        self.samples = []
        rollout_steps = 20

        mapping = pd.read_csv(mapping_file)
        for row in tqdm(mapping.itertuples(index=False), total=len(mapping)):
            filename = row[0]
            p = row[1]

            df = pd.read_csv(DATASETS_DIR / "WDN/simple_transformation_set_3" / filename)
            df = resample_counts(df)

            if df.isna().any().any():
                print(filename, "resulted in nan")

            A = df["A"]
            B = df["B"]

            """
            node features:
                x              y
            A:  [A(t)]      -> [A(t+dt), A(t+2*dt), ..., A(t+rollout_steps*dt)]
            B:  [B(t)]      -> [B(t+dt), B(t+2*dt), ..., B(t+rollout_steps*dt)]
            """
            for i in range(len(df) - rollout_steps):
                graph = create_graph(A.iloc[i], B.iloc[i], p)
                graph.y = torch.tensor([
                    A.iloc[i + 1:i + rollout_steps + 1].to_numpy(),
                    B.iloc[i + 1:i + rollout_steps + 1].to_numpy()
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

    mapping_file = DATASETS_DIR / "WDN/simple_transformation_set_3.csv"
    dataset = ReactionDataset(mapping_file)

    logger.info(f"Saving dataset to {DATASET_PATH}")
    torch.save(dataset, DATASET_PATH)
    return dataset


class ReactionGNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(3, 64),   # [source_count, target_count, p]
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        """
        With batch_size=32 we will have the following shapes:
        - data.x: [ [A1], [B1], [A2], [B2], ..., [A32], [B32] ]: 32 values for each species count
        - data.edge_index: [ [0, 2, ..., 62 ], [1, 3, ..., 63] ]: Total of 64 values mapping to data.x left side (A) and right side (B)
        - data.edge_attr: [ [p1], [p2], ..., [p32] ]: 32 values of p
        """
        x = data.x                   # [ [A], [B] ]: Counts for each node
        src, dst = data.edge_index   # [ [0], [1] ]: Single edge from A to B
        p = data.edge_attr

        # we get [ [source_counts], [target_counts], p]
        edge_input = torch.cat([
            x[src],
            x[dst],
            p
        ], dim=-1)

        frac = torch.sigmoid(self.edge_mlp(edge_input))   # let mlp predict fraction between 0 and 1
        # calculate how much A we lose and how much B we gain
        # the LLM wrote code that multiplied with p but that doesn't make sense for dt > 1 for example

        # for all cases of p being 1 or 0 set flux to x[src] (all transform) or 0, otherwise flux is just 
        # frac * x[src]
        flux = frac * p * x[src]

        incoming = torch.zeros_like(x)
        outgoing = torch.zeros_like(x)

        incoming.index_add_(0, dst, flux)
        outgoing.index_add_(0, src, flux)

        # add changes to current counts
        x_next = x - outgoing + incoming
        return x_next


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


def predict_next_step(model, A, B, p, device="cpu"):
    graph = create_graph(A, B, p).to(device)

    with torch.no_grad():
        pred = model(graph).squeeze(-1)
        A_next, B_next = pred.cpu().tolist()

    return A_next, B_next


def predict_trajectory(model, A0, B0, p, steps, dt=0.05, device="cpu"):
    A = float(A0)
    B = float(B0)
    trajectory = [(0.0, A, B)]

    with torch.no_grad():
        for step in range(1, steps + 1):
            graph = create_graph(A, B, p).to(device)
            pred = model(graph).squeeze(-1)
            A, B = pred.cpu().tolist()
            trajectory.append((step * dt, A, B))

    return trajectory


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not MODEL_PATH.exists():
        train(device=device, epochs=5)
    
    model = load_model(device)

    A0 = 320
    B0 = 80
    N = A0 + B0
    p = 0.05

    trajectory = predict_trajectory(
        model=model,
        A0=A0 / N,
        B0=B0 / N,
        p=p,
        steps=int(60 / 0.05),
        dt=0.05,
        device=device
    )
    trajectory = np.array(trajectory)

    plt.plot(trajectory[:, 0], trajectory[:, 1] * N, label="A")
    plt.plot(trajectory[:, 0], trajectory[:, 2] * N, label="B")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()
    