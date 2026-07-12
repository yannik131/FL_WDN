import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from util.paths import DATASETS_DIR
import flwr as fl

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

N_CLIENTS = 1
SEED = 42
df = pd.read_csv(DATASETS_DIR / "FL/lv_heat_map_full_3.csv")
X = df[["p1","p2","p3","p4","p5","p6"]].to_numpy()
y = df["is_lv"].astype(np.float32).to_numpy()
train_idx = np.load(DATASETS_DIR / "FL/train_idx_3.npy")
test_idx = np.load(DATASETS_DIR / "FL/test_idx_3.npy")
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def split_dataset_randomly():
    client_datasets = []
    groups = pd.factorize(pd.DataFrame(X[train_idx]).apply(tuple, axis=1))[0]
    sgkf = StratifiedGroupKFold(
        n_splits=N_CLIENTS,
        shuffle=True,
        random_state=SEED
    )
    # split returns train_idx, test_idx
    # we get N_CLIENTS splits where test_idx contains 1/N_CLIENTS of the data
    # we ignore train_idx to "abuse" this to get N_CLIENTS stratified group splits
    for _, idx in sgkf.split(X_train, y_train, groups):
        client_datasets.append((X_train[idx], y_train[idx]))

    return client_datasets

def split_dataset_cohesively():
    # bounds of p_i used for dataset creation
    lower = np.array([0, 0.0, 0.0, 0.8, 0, 0])
    upper = np.array([1.0, 0.2, 0.15, 1.0, 0.11, 0.11])
    X_raw = X[train_idx]
    client_datasets = []
    for N in range(N_CLIENTS):
        lower_factor = N / N_CLIENTS
        upper_factor = (N + 1) / N_CLIENTS
        mins = lower + lower_factor * (upper - lower)
        maxs = lower + upper_factor * (upper - lower)

        mask = (X_raw >= mins) & (X_raw <= maxs)
        idx = np.where(np.all(mask, axis=1))[0]
        client_datasets.append((X_train[idx], y_train[idx]))
    return client_datasets

client_datasets = split_dataset_cohesively()

# return all model parameters
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

global_model = MLP()

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=N_CLIENTS,
    fraction_evaluate=0,
    min_evaluate_clients=0,
    evaluate_fn=None,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(global_model))
)

def client_fn(cid: str):
    X, y = client_datasets[int(cid)]
    return FlowerClient(X, y).to_client()

history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=N_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
    client_resources={'num_cpus': 1},
)

final_parameters = fl.common.parameters_to_ndarrays(strategy.latest_parameters)
set_parameters(global_model, final_parameters)
p = predict_proba(global_model, X_test)

print("Brier score:", brier_score_loss(y_test, p))
print("Log loss:", log_loss(y_test, p))
print("ROC AUC:", roc_auc_score(y_test, p))