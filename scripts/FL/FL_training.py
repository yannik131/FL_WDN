import random
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from util.paths import DATASETS_DIR, RESULTS_DIR


# =========================
# Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# Config
# =========================
NUM_CLIENTS = 4
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-3  # similar role to alpha in sklearn MLP

DATA_PATH = DATASETS_DIR / "FL/lv_heat_map_full_2_df.csv"
BASELINE_MODEL_PATH = RESULTS_DIR / "FL/full_lv_model.joblib"
FL_MODEL_PATH = RESULTS_DIR / "FL/full_lv_model_fedavg.pt"


# =========================
# Data
# =========================
df = pd.read_csv(DATA_PATH)

feature_cols = ["p1", "p2", "p3", "p4", "p5", "p6"]
X = df[feature_cols].to_numpy(dtype=np.float32)
y = df["is_lv"].astype(int).to_numpy()

# Same split logic as baseline
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=SEED,
)

print("Global train class counts:", np.unique(y_train, return_counts=True))
print("Global test class counts: ", np.unique(y_test, return_counts=True))

# Use one global scaler for all clients and test data
# (simple and fair for comparison; stricter FL would federate normalization too)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)


def stratified_client_split(X, y, num_clients, seed=42):
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        parts = np.array_split(idx, num_clients)
        for cid, part in enumerate(parts):
            client_indices[cid].extend(part.tolist())

    out = []
    for cid in range(num_clients):
        idx = np.array(client_indices[cid], dtype=int)
        rng.shuffle(idx)
        out.append((X[idx], y[idx]))
    return out


client_partitions = stratified_client_split(X_train, y_train, NUM_CLIENTS, seed=SEED)

for cid, (_, yc) in enumerate(client_partitions):
    print(f"Client {cid} class counts:", np.unique(yc, return_counts=True))


# =========================
# Model
# =========================
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


def get_parameters(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train_local(model, X, y, epochs, batch_size, lr, weight_decay):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
    return probs


def compute_metrics(y_true, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "roc_auc": roc_auc_score(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "log_loss": log_loss(y_true, p),
    }


# =========================
# Flower client
# =========================
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X_local, y_local):
        self.X_local = X_local
        self.y_local = y_local
        self.model = MLP(in_dim=X_local.shape[1])

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train_local(
            self.model,
            self.X_local,
            self.y_local,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        return get_parameters(self.model), len(self.X_local), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        p = predict_proba(self.model, self.X_local)
        m = compute_metrics(self.y_local, p)
        return float(m["log_loss"]), len(self.X_local), m


def client_fn(cid: str):
    X_local, y_local = client_partitions[int(cid)]
    return FlowerClient(X_local, y_local).to_client()


# =========================
# Server-side evaluation
# =========================
global_model = MLP(in_dim=X_train.shape[1])


def evaluate_fn(server_round, parameters, config):
    set_parameters(global_model, parameters)
    p = predict_proba(global_model, X_test)
    m = compute_metrics(y_test, p)
    print(
        f"[Round {server_round}] "
        f"AUC={m['roc_auc']:.6f}, "
        f"Brier={m['brier']:.6f}, "
        f"LogLoss={m['log_loss']:.6f}"
    )
    return float(m["log_loss"]), m


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics


strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    fraction_evaluate=0.0,
    min_evaluate_clients=0,
    evaluate_fn=evaluate_fn,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(global_model)),
)

# =========================
# Run FL simulation
# =========================
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_cpus": 1},
)

# Load final aggregated weights into global model
final_parameters = fl.common.parameters_to_ndarrays(strategy.latest_parameters)
set_parameters(global_model, final_parameters)

# Final FL metrics
p_fl = predict_proba(global_model, X_test)
fl_metrics = compute_metrics(y_test, p_fl)

print("\nFinal FL model")
print("ROC AUC:   ", fl_metrics["roc_auc"])
print("Brier:     ", fl_metrics["brier"])
print("Log loss:  ", fl_metrics["log_loss"])

# Save FL model
torch.save(
    {
        "state_dict": global_model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_cols": feature_cols,
    },
    FL_MODEL_PATH,
)

# =========================
# Baseline comparison
# =========================
baseline_model = joblib.load(BASELINE_MODEL_PATH)
p_baseline = baseline_model.predict_proba(
    df.loc[X_test.shape[0] * 0 : 0, feature_cols]
)  # placeholder
