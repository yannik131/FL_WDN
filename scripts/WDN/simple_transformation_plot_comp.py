from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from WDN.train_simple_gnn import ReactionGNN
from util.paths import RESULTS_DIR
import json
from util.paths import CONFIG_DIR, DATASETS_DIR
from util.task import Task, execute_tasks, create_mapfile
from itertools import product

MODEL_PATH = RESULTS_DIR / "WDN/simple_gnn_transformation_3.pt"

def run_reaction_1000_times(A0, B0, p, set_name):
    mapfile_path = DATASETS_DIR / f'WDN/{set_name}.csv'
    output_dir = DATASETS_DIR / f'WDN/{set_name}/'
    mapping = {
        "p": ["config", "reactions", 0, "probability"],
        "N": ["config", "cellMembraneType", "discCount"],
        "f_A": ["config", "cellMembraneType", "discTypeDistribution", "A"],
        "f_B": ["config", "cellMembraneType", "discTypeDistribution", "B"]
    }
    N = A0 + B0

    if len(list(output_dir.glob('*.csv'))) == 1000:
        return

    tasks = []
    for i in range(1000):
        params = dict(
            filename=f"{set_name}_{i:04d}.csv",
            r=i,
            p=p,
            N=N,
            f_A=A0/N if N > 0 else 1,
            f_B=B0/N if N > 0 else 0
        )
        tasks.append(Task(params, mapping))

    create_mapfile(tasks, mapfile_path)

    with open(CONFIG_DIR / "WDN/transformation_simple.json") as f:
        cfg = json.load(f)

    execute_tasks(tasks, cfg, output_dir)

def get_average_trajectory(A0, B0, p):
    set_name = 'transformation_reaction'
    run_reaction_1000_times(A0, B0, p, set_name)
    set_dir = DATASETS_DIR / f'WDN/{set_name}/'
    dfs = [pd.read_csv(file) for file in set_dir.glob('*.csv')]
    df = pd.concat(dfs, ignore_index=True)
    average = df.groupby("ElapsedTime[s]")[["A", "B"]].mean()

    return average.index.to_numpy(), average["A"].to_numpy(), average["B"].to_numpy()

def get_analytical_trajectory(A0, B0, p):
    dt = 0.05
    t = np.arange(0, 60 + dt, dt)
    B = B0 + A0 * (1 - (1 - p)**t)
    A = (A0 + B0) - B

    return t, A, B 
    

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

def load_model(device="cpu"):
    model = ReactionGNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_predicted_trajectory(A0, B0, p):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    N = A0 + B0

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

    return trajectory[:, 0], trajectory[:, 1] * N, trajectory[:, 2] * N

def plot_trajectory(t, A, B, ax, label_suffix, c, style):
    ax.plot(t, A, style, color=c, label=label_suffix)
    ax.plot(t, B, style, color=c)

if __name__ == '__main__':
    A0 = 500
    B0 = 50
    p = 0.07
    fig, ax = plt.subplots()
    plot_trajectory(*get_average_trajectory(A0, B0, p), ax, "averaged", "blue", "--")
    plot_trajectory(*get_predicted_trajectory(A0, B0, p), ax, "predicted", "green", ":")
    plot_trajectory(*get_analytical_trajectory(A0, B0, p), ax, "analytical", "red", "-")
    plt.legend()
    plt.show()
