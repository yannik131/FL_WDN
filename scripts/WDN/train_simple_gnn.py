import torch 
from torch_geometric.data import Data, Dataset
import pandas as pd
from util.paths import DATASETS_DIR
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def resample_counts(df, dt=0.05):
    time_col = "ElapsedTime[s]"
    x = df[time_col]
    new_time = np.arange(x.min(), x.max() + dt, dt)
    count_cols = df.columns.drop(time_col)

    df_interp = pd.DataFrame({time_col: new_time})
    for col in count_cols:
        df_interp[col] = np.interp(
            new_time,
            x,
            df[col]
        )
    df_interp[count_cols] = df_interp[count_cols].round().astype(int)
    return df_interp

def create_graph(A, B, p):
    x = torch.tensor([
        [A],
        [B],
    ], dtype=torch.float)

    edge_index = torch.tensor([
        [0],
        [1]
    ], dtype=torch.long)

    edge_attr = torch.tensor([[p]], dtype=torch.float)

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
            df = pd.read_csv(DATASETS_DIR / "WDN/simple_transformation_set_2/" / filename)
            df = resample_counts(df)
            A = df["A"]
            B = df["B"]
            for i in range(len(df) - 1):
                graph = create_graph(
                    A.iloc[i],
                    B.iloc[i],
                    p
                )
                target = torch.tensor([[
                    A.iloc[i+1],
                    B.iloc[i+1]
                ]], dtype=torch.float)
                graph.y = target 
                self.samples.append(graph)

    def len(self):
        return len(self.samples)
    
    def get(self, idx):
        return self.samples[idx]
    
def load_dataset():
    dataset_file = DATASETS_DIR / "WDN/simple_transformation_data.pt"
    if not dataset_file.exists():
        mapping_file = DATASETS_DIR / "WDN/simple_transformation_set_2.csv"
        dataset = ReactionDataset(mapping_file)
        logger.info("Saving dataset")
        torch.save(dataset, dataset_file)
        logger.info("Saved to ", dataset_file)
    else:
        dataset = torch.load(dataset_file, weights_only=False)
    return dataset
            
class ReactionGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 32)
        self.conv2 = GCNConv(32, 32)
        self.edge_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )
        self.out = nn.Linear(32, 2)

    def forward(self, data):
        x = data.x
        x = self.conv1(x, data.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, data.edge_index)
        x = torch.relu(x)

        p = self.edge_net(data.edge_attr)
        target_nodes = data.edge_index[1]
        x[target_nodes] = x[target_nodes] + p
        return self.out(x)[target_nodes]

def train():
    dataset = load_dataset()
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    model = ReactionGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    print(f"Number of batches: {len(loader)}")
    
    for epoch in range(10):
        total_loss = 0
        for i, batch in enumerate(loader):
            print(f"Batch {i}/{len(loader)}")
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: {total_loss}")

train()
