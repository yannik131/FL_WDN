import random
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


SEED = 42
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-3


def train_federated_model(
    dataset_path,
    random_split_across_clients,
    num_clients,
    output_model_path,
):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset_path = Path(dataset_path)
    output_model_path = Path(output_model_path)

    feature_cols = ["p1", "p2", "p3", "p4", "p5", "p6"]
    target_col = "is_lv"
    group_col = "__group_id__"

    df = pd.read_csv(dataset_path).copy()
    df[group_col] = df.groupby(feature_cols, sort=False).ngroup()

    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df[target_col].astype(int).to_numpy()
    groups_all = df[group_col].to_numpy()

    train_idx, test_idx = _stratified_group_train_test_split(
        X_all,
        y_all,
        groups_all,
        test_size=0.2,
        seed=SEED,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_col].astype(int).to_numpy()
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df[target_col].astype(int).to_numpy()

    print("Global train class counts:", np.unique(y_train, return_counts=True))
    print("Global test class counts: ", np.unique(y_test, return_counts=True))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    if random_split_across_clients:
        client_partitions = _stratified_group_client_split(
            X_train,
            y_train,
            train_df[group_col].to_numpy(),
            num_clients=num_clients,
            seed=SEED,
        )
    else:
        client_partitions = _cohesive_region_client_split(
            train_df=train_df,
            X_train_scaled=X_train,
            y_train=y_train,
            feature_cols=feature_cols,
            group_col=group_col,
            num_clients=num_clients,
        )

    for cid, (_, yc) in enumerate(client_partitions):
        print(f"Client {cid} class counts:", np.unique(yc, return_counts=True))

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
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

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
        metrics = {
            "brier": brier_score_loss(y_true, p),
            "log_loss": log_loss(y_true, p),
        }
        metrics["roc_auc"] = (
            roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan")
        )
        return metrics

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
                server_round,
                results,
                failures,
            )
            if aggregated_parameters is not None:
                self.latest_parameters = aggregated_parameters
            return aggregated_parameters, aggregated_metrics

    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_evaluate=0.0,
        min_evaluate_clients=0,
        evaluate_fn=evaluate_fn,
        initial_parameters=fl.common.ndarrays_to_parameters(
            get_parameters(global_model)
        ),
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    if strategy.latest_parameters is None:
        raise RuntimeError("No aggregated model parameters were produced.")

    final_parameters = fl.common.parameters_to_ndarrays(strategy.latest_parameters)
    set_parameters(global_model, final_parameters)

    p_fl = predict_proba(global_model, X_test)
    fl_metrics = compute_metrics(y_test, p_fl)

    print("\nFinal FL model")
    print("ROC AUC:   ", fl_metrics["roc_auc"])
    print("Brier:     ", fl_metrics["brier"])
    print("Log loss:  ", fl_metrics["log_loss"])

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": global_model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_cols": feature_cols,
        },
        output_model_path,
    )

    return {
        "history": history,
        "metrics": fl_metrics,
        "model_path": output_model_path,
    }


def _stratified_group_train_test_split(X, y, groups, test_size=0.2, seed=42):
    n_splits = round(1 / test_size)
    if not np.isclose(test_size, 1 / n_splits):
        raise ValueError(
            "`test_size` must be the reciprocal of an integer for "
            "`StratifiedGroupKFold`."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return train_idx, test_idx


def _stratified_group_client_split(X, y, groups, num_clients, seed=42):
    splitter = StratifiedGroupKFold(
        n_splits=num_clients,
        shuffle=True,
        random_state=seed,
    )

    partitions = []
    for _, client_idx in splitter.split(X, y, groups):
        partitions.append((X[client_idx], y[client_idx]))

    return partitions


def _cohesive_region_client_split(
    train_df,
    X_train_scaled,
    y_train,
    feature_cols,
    group_col,
    num_clients,
):
    work_df = train_df.copy()
    work_df["__row_idx__"] = np.arange(len(work_df))

    group_features = work_df.groupby(group_col, sort=False)[feature_cols].first()
    group_values = group_features.to_numpy(dtype=np.float32)

    bucket_matrix = np.floor(group_values * num_clients).astype(int)
    bucket_matrix = np.clip(bucket_matrix, 0, num_clients - 1)

    same_bucket = np.all(bucket_matrix == bucket_matrix[:, [0]], axis=1)
    if not np.all(same_bucket):
        bad_groups = group_features.index[~same_bucket].tolist()[:10]
        raise ValueError(
            "Cohesive-region split is ambiguous for some groups: at least one "
            "group has `p1`-`p6` falling into different client ranges. "
            f"Example group ids: {bad_groups}"
        )

    group_to_client = pd.Series(bucket_matrix[:, 0], index=group_features.index)
    work_df["__client_id__"] = work_df[group_col].map(group_to_client)

    partitions = []
    for cid in range(num_clients):
        idx = work_df.loc[work_df["__client_id__"] == cid, "__row_idx__"].to_numpy()
        if len(idx) == 0:
            raise ValueError(
                f"Client {cid} received 0 samples in cohesive-region split."
            )
        partitions.append((X_train_scaled[idx], y_train[idx]))

    return partitions
