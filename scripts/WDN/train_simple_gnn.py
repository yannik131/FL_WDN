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

MODEL_PATH = RESULTS_DIR / "WDN/simple_gnn_flux.pt"
DATASET_PATH = DATASETS_DIR / "WDN/simple_transformation_data_flux.pt"


def resample_counts(df, dt=0.05):
    time_col = "ElapsedTime[s]"
    x = df[time_col].to_numpy()
    new_time = np.arange(x.min(), x.max() + dt, dt)
    count_cols = df.columns.drop(time_col)

    df_interp = pd.DataFrame({time_col: new_time})
    for col in count_cols:
        df_interp[col] = np.interp(new_time, x, df[col].to_numpy())

    return df_interp


def create_graph(A, B, p):
    x = torch.tensor([
        [float(A)],
        [float(B)],
    ], dtype=torch.float)

    edge_index = torch.tensor([
        [0],
        [1],
    ], dtype=torch.long)

    edge_attr = torch.tensor([
        [float(p)]
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

        mapping = pd.read_csv(mapping_file)
        for row in tqdm(mapping.itertuples(index=False), total=len(mapping)):
            filename = row[0]
            p = row[1]

            df = pd.read_csv(DATASETS_DIR / "WDN/simple_transformation_set_3" / filename)
            df = resample_counts(df)

            A = df["A"]
            B = df["B"]

            for i in range(len(df) - 1):
                graph = create_graph(A.iloc[i], B.iloc[i], p)
                graph.y = torch.tensor([
                    [float(A.iloc[i + 1])],
                    [float(B.iloc[i + 1])],
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

    mapping_file = DATASETS_DIR / "WDN/simple_transformation_set_2.csv"
    dataset = ReactionDataset(mapping_file)

    logger.info(f"Saving dataset to {DATASET_PATH}")
    torch.save(dataset, DATASET_PATH)
    return dataset


class ReactionGNN(nn.Module):
    def __init__(self, max_count=320.0):
        super().__init__()
        self.max_count = max_count

        self.edge_mlp = nn.Sequential(
            nn.Linear(3, 64),   # [source_count, target_count, p]
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x = data.x                   # [num_nodes, 1]
        src, dst = data.edge_index   # [num_edges], [num_edges]

        edge_input = torch.cat([
            x[src] / self.max_count,
            x[dst] / self.max_count,
            data.edge_attr
        ], dim=-1)

        frac = torch.sigmoid(self.edge_mlp(edge_input))   # [num_edges, 1], in [0, 1]
        flux = frac * data.edge_attr * x[src]             # [num_edges, 1]

        incoming = torch.zeros_like(x)
        outgoing = torch.zeros_like(x)

        incoming.index_add_(0, dst, flux)
        outgoing.index_add_(0, src, flux)

        x_next = x - outgoing + incoming
        return x_next


def train(device="cpu", epochs=200):
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
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}")

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
        train(device=device, epochs=40)
    
    """
    model = load_model(device)

    A0 = 750
    B0 = 144
    p = 0.05

    trajectory = predict_trajectory(
        model=model,
        A0=A0,
        B0=B0,
        p=p,
        steps=int(60 / 0.05),
        dt=0.05,
        device=device
    )
    trajectory = np.array(trajectory)

    plt.plot(trajectory[:, 0], trajectory[:, 1], label="A")
    plt.plot(trajectory[:, 0], trajectory[:, 2], label="B")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
