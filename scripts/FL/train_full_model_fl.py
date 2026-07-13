import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from util.paths import DATASETS_DIR, RESULTS_DIR
import flwr as fl
from dataclasses import dataclass
from pathlib import Path
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

FEATURE_COLS = ["p1","p2","p3","p4","p5","p6"]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

@dataclass
class Dataset:
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    dataset_path: Path

def split_dataset_randomly(n_clients, d: Dataset):
    if n_clients == 1:
        return [(d.X_train_scaled, d.y_train)]

    client_datasets = []
    groups = pd.factorize(pd.DataFrame(d.X_train_raw).apply(tuple, axis=1))[0]
    sgkf = StratifiedGroupKFold(
        n_splits=n_clients,
        shuffle=True,
        random_state=SEED
    )
    # split returns train_idx, test_idx
    # we get N_CLIENTS splits where test_idx contains 1/N_CLIENTS of the data
    # we ignore train_idx to "abuse" this to get N_CLIENTS stratified group splits
    for _, idx in sgkf.split(d.X_train_scaled, d.y_train, groups):
        client_datasets.append((d.X_train_scaled[idx], d.y_train[idx]))

    return client_datasets

def split_dataset_cohesively(n_clients, d: Dataset):
    # split cohesively along p1 axis since it runs from 0 to 1
    client_datasets = []
    for N in range(n_clients):
        lower = N / n_clients
        upper = (N + 1) / n_clients

        mask = (d.X_train_raw[:, 0] >= lower) & (d.X_train_raw[:, 0] <= upper)
        idx = np.where(mask)[0]
        client_datasets.append((d.X_train_scaled[idx], d.y_train[idx]))
    return client_datasets

def get_parameters(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor).cpu().numpy()
        probs = 1.0 / (1 + np.exp(-logits))
    return probs

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        self.model = MLP()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train_local(self.model, self.X_train, self.y_train)
        return get_parameters(self.model), len(self.X_train), {}

def train_local(model, X, y):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    for _ in range(5):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

def read_dataset(dataset_path, train_idx_path, test_idx_path):
    df = pd.read_csv(dataset_path)
    X = df[FEATURE_COLS].to_numpy()
    y = df["is_lv"].astype(np.float32).to_numpy()
    train_idx = np.load(train_idx_path)
    test_idx = np.load(test_idx_path)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    d = Dataset(
        X_train_raw=X_train,
        X_test_raw=X_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        dataset_path=dataset_path
    )

    return d

def train_fl_model(n_clients, client_datasets):
    global_model = MLP()
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=n_clients,
        fraction_evaluate=0,
        min_evaluate_clients=0,
        evaluate_fn=None,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(global_model))
    )

    fl.simulation.start_simulation(
        client_fn=lambda cid: FlowerClient(*client_datasets[int(cid)]).to_client(),
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources={'num_cpus': 1},
    )

    final_parameters = fl.common.parameters_to_ndarrays(strategy.latest_parameters)
    set_parameters(global_model, final_parameters)

    return global_model

def save_model(n_clients, split_random, d: Dataset, model):
    p = predict_proba(model, d.X_test_scaled)

    scores_csv = RESULTS_DIR / "FL/scores.csv"
    if not scores_csv.exists():
        with open(scores_csv, "w") as file:
            file.write("model_id,n_clients,split_random,dataset_filename,brier,log,auc\n")

    with open(scores_csv) as file:
        model_id = sum(1 for _ in file) - 1

    brier_score = brier_score_loss(d.y_test, p)
    log_score = log_loss(d.y_test, p)
    auc_score = roc_auc_score(d.y_test, p)

    with open(scores_csv, "a") as file:
        file.write(",".join(map(str, [model_id, n_clients, int(split_random), d.dataset_path.name, brier_score, log_score, auc_score])))
        file.write("\n")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler_mean": d.scaler.mean_,
            "scaler_scale": d.scaler.scale_,
            "feature_cols": FEATURE_COLS,
        },
        RESULTS_DIR / f"FL/model_{model_id}.pt",
    )

def train_fl(n_clients, split_random, dataset_path, train_idx_path, test_idx_path):
    d = read_dataset(dataset_path, train_idx_path, test_idx_path)

    if split_random:
        client_datasets = split_dataset_randomly(n_clients, d)
    else:
        client_datasets = split_dataset_cohesively(n_clients, d)

    model = train_fl_model(n_clients, client_datasets)
    save_model(n_clients, split_random, d, model)

for n_clients in [32, 64]:
    for split_random in [True, False]:
        dataset_path = DATASETS_DIR / "FL/lv_heat_map_full_3.csv"
        train_idx_path = DATASETS_DIR / "FL/train_idx_3.npy"
        test_idx_path = DATASETS_DIR / "FL/test_idx_3.npy"

        print(f"Training with n_clients={n_clients}, split_random={split_random}")
        train_fl(n_clients, split_random, dataset_path, train_idx_path, test_idx_path)