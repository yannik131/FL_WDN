import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from util.paths import DATASETS_DIR, RESULTS_DIR
import numpy as np
import matplotlib.ticker as mtick
import torch
import torch.nn as nn

def set_ticks(ax, n):
    ticks = np.arange(0, 1.01, 0.05)
    tick_labels = [f"{t:.2f}" for t in ticks]
    pos = ticks * (n - 1)
    ax.set_xticks(pos)
    ax.set_yticks(pos)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

heatmap_df_path = DATASETS_DIR / "FL/lv_heat_map_simple_prom_150_120.csv"

df_all = pd.read_csv(heatmap_df_path)

heat = df_all.groupby(["p1", "p2"])["is_lv"].mean().reset_index()
pivot = heat.pivot(index="p1", columns="p2", values="is_lv") * 100

class MLP(nn.Module):
    def __init__(self, in_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

scores_df = pd.read_csv(RESULTS_DIR / "FL/scores.csv")
for row in scores_df.itertuples():
    model_id = row.model_id
    n_clients = row.n_clients
    split_random = row.split_random

    checkpoint = torch.load(
        RESULTS_DIR / f"FL/model_{model_id}.pt",
        map_location="cpu",
        weights_only=False,
    )

    model = MLP(in_dim=len(checkpoint["feature_cols"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scaler_mean = np.array(checkpoint["scaler_mean"])
    scaler_scale = np.array(checkpoint["scaler_scale"])
    feature_cols = checkpoint["feature_cols"]

    # 1000 x 1000 prediction grid
    p = np.arange(0, 1, 0.001)
    p1_grid, p2_grid = np.meshgrid(p, p, indexing="ij")

    grid_model = pd.DataFrame({
        "p1": p1_grid.ravel(),
        "p2": p2_grid.ravel(),
        "p3": 0.05,
        "p4": 0.9,
        "p5": 0.01,
        "p6": 0.01
    })

    # Use exactly the same preprocessing as training
    X_grid = grid_model[feature_cols].to_numpy(dtype=np.float32)
    X_grid = (X_grid - scaler_mean) / scaler_scale

    X_grid = torch.tensor(
        X_grid,
        dtype=torch.float32
    )

    # Predict
    with torch.no_grad():
        logits = model(X_grid)
        probs = torch.sigmoid(logits)

    grid_model["lv_prob_model"] = probs.numpy()

    pivot_model = (
        grid_model
        .pivot(index="p1", columns="p2", values="lv_prob_model")
        * 100
    )

    fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 6))

    sns.heatmap(
        pivot,
        cmap="viridis",
        vmin=0,
        vmax=100,
        square=True,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title("Simulated")

    sns.heatmap(
        pivot_model,
        cmap="viridis",
        vmin=0,
        vmax=100,
        square=True,
        ax=axes[1],
        cbar_kws={'format': mtick.PercentFormatter(100)}
    )
    axes[1].set_title("Predicted")

    set_ticks(axes[0], len(pivot.columns))
    set_ticks(axes[1], len(pivot_model.columns))

    # plt.savefig(DATASETS_DIR / f"FL/fl_comp_plots/fl_rand_comp_{n_clients}_{split_random}.jpg", dpi=300)
    plt.close()

    print(f"N = {n_clients}, split_random = {split_random}")
    unique_probs = np.sort(grid_model["lv_prob_model"].round(2).unique())
    print(unique_probs)
    input()
