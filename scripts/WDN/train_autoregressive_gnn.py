import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv


logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Random seed set to %d", seed)


def parse_reaction(reaction_str: str):
    """
    Supports:
    - A->B
    - A->B+C
    - A+B->C
    - A+B->C+D
    """
    left, right = reaction_str.split("->")
    reactants = [x.strip() for x in left.split("+") if x.strip()]
    products = [x.strip() for x in right.split("+") if x.strip()]
    return reactants, products


def infer_columns(df: pd.DataFrame):
    reserved = {"filename", "r", "repetition", "repeat", "rep", "seed"}
    n_col = "N" if "N" in df.columns else None
    frac_cols = [c for c in df.columns if c.startswith("f_")]
    reaction_cols = [c for c in df.columns if c not in reserved and c != n_col and c not in frac_cols]
    if "filename" not in df.columns:
        raise ValueError("Mapping CSV must contain a `filename` column.")
    return n_col, frac_cols, reaction_cols


class ReactionCountDataset(Dataset):
    """
    Lazy dataset:
    - builds only an index at startup
    - loads one scenario CSV on demand
    - builds graph objects in `__getitem__`
    """

    def __init__(self, mapping_csv, series_dir, config_json, log_interval_scenarios=50):
        self.mapping_csv = Path(mapping_csv)
        self.series_dir = Path(series_dir)
        self.config_json = Path(config_json)
        self.log_interval_scenarios = max(int(log_interval_scenarios), 1)

        logger.info("Loading config from %s", self.config_json)
        with open(self.config_json, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        logger.info("Loading mapping CSV from %s", self.mapping_csv)
        self.mapping = pd.read_csv(self.mapping_csv)
        self.n_col, self.frac_cols, self.reaction_cols = infer_columns(self.mapping)

        logger.info(
            "Mapping loaded: %d rows | N column: %s | fraction columns: %d | reaction columns: %d",
            len(self.mapping),
            self.n_col,
            len(self.frac_cols),
            len(self.reaction_cols),
        )

        disc_cfg = self.config["config"]["discTypes"]
        self.disc_names = [d["name"] for d in disc_cfg]
        self.masses = [float(d["mass"]) for d in disc_cfg]
        self.radii = [float(d["radius"]) for d in disc_cfg]
        self.disc_index = {name: i for i, name in enumerate(self.disc_names)}

        self.default_N = float(self.config["config"]["cellMembraneType"]["discCount"])
        self.max_N = float(self.mapping[self.n_col].max()) if self.n_col else self.default_N
        self.max_N = max(self.max_N, 1.0)

        self.mass_scale = max(max(self.masses), 1.0)
        self.radius_scale = max(max(self.radii), 1.0)

        self.dt = 1e-3
        self.feature_dim = 7  # [count_norm, mass, radius, reaction_prob, is_disc, is_reaction, N_norm]

        self.num_disc = len(self.disc_names)
        self.num_rxn = len(self.reaction_cols)
        self.num_nodes = self.num_disc + self.num_rxn

        self.mass_tensor = torch.tensor(
            [m / self.mass_scale for m in self.masses], dtype=torch.float32
        )
        self.radius_tensor = torch.tensor(
            [r / self.radius_scale for r in self.radii], dtype=torch.float32
        )

        self.meta = {
            "disc_names": self.disc_names,
            "disc_static": [
                {"name": n, "mass": m, "radius": r}
                for n, m, r in zip(self.disc_names, self.masses, self.radii)
            ],
            "reaction_cols": self.reaction_cols,
            "max_N": self.max_N,
            "mass_scale": self.mass_scale,
            "radius_scale": self.radius_scale,
            "dt": self.dt,
            "feature_dim": self.feature_dim,
        }

        logger.info(
            "Dataset config: %d disc types | %d reaction nodes | %d total nodes",
            self.num_disc,
            self.num_rxn,
            self.num_nodes,
        )

        logger.info("Building static graph...")
        self._build_static_graph()

        logger.info("Building sample index...")
        self._build_index()

        self._cache_scenario_id = None
        self._cache_counts_norm = None

    def _reaction_prob_vector_from_row(self, row):
        probs = np.zeros(self.num_rxn, dtype=np.float32)
        for i, rcol in enumerate(self.reaction_cols):
            val = row[rcol]
            probs[i] = 0.0 if pd.isna(val) else float(val)
        return probs

    def _build_static_graph(self):
        edge_src, edge_dst, edge_attr = [], [], []

        for j, rstr in enumerate(self.reaction_cols):
            node_idx = self.num_disc + j
            reactants, products = parse_reaction(rstr)

            for name in reactants:
                if name not in self.disc_index:
                    raise ValueError(f"Reactant `{name}` not found in config disc types.")
                d_idx = self.disc_index[name]
                edge_src += [d_idx, node_idx]
                edge_dst += [node_idx, d_idx]
                edge_attr += [[1.0, 0.0], [1.0, 0.0]]

            for name in products:
                if name not in self.disc_index:
                    raise ValueError(f"Product `{name}` not found in config disc types.")
                d_idx = self.disc_index[name]
                edge_src += [node_idx, d_idx]
                edge_dst += [d_idx, node_idx]
                edge_attr += [[0.0, 1.0], [0.0, 1.0]]

        if len(edge_src) == 0:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.edge_attr = torch.empty((0, 2), dtype=torch.float32)
        else:
            self.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            self.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        self.loss_mask = torch.tensor(
            [True] * self.num_disc + [False] * self.num_rxn, dtype=torch.bool
        )

        logger.info(
            "Static graph built: %d edges | loss_mask disc nodes: %d",
            self.edge_index.shape[1],
            int(self.loss_mask.sum().item()),
        )

    def _read_num_rows(self, csv_path):
        return pd.read_csv(csv_path, usecols=[0]).shape[0]

    def _build_index(self):
        self.scenarios = []
        offsets = [0]
        total_rows = len(self.mapping)

        logger.info("Indexing %d scenario files...", total_rows)

        for i, (_, row) in enumerate(self.mapping.iterrows(), start=1):
            ts_path = self.series_dir / row["filename"]
            num_rows = self._read_num_rows(ts_path)
            num_samples = max(num_rows - 1, 0)

            N = float(row[self.n_col]) if self.n_col else self.default_N
            N = max(N, 1.0)

            self.scenarios.append(
                {
                    "path": ts_path,
                    "N": N,
                    "reaction_probs": self._reaction_prob_vector_from_row(row),
                    "num_samples": num_samples,
                }
            )
            offsets.append(offsets[-1] + num_samples)

            if i % self.log_interval_scenarios == 0 or i == total_rows:
                logger.info(
                    "Indexed %d/%d scenarios | latest file: %s | rows: %d | cumulative samples: %d",
                    i,
                    total_rows,
                    ts_path.name,
                    num_rows,
                    offsets[-1],
                )

        self.sample_offsets = np.array(offsets, dtype=np.int64)

        logger.info(
            "Finished indexing: %d scenarios | %d total samples",
            len(self.scenarios),
            int(self.sample_offsets[-1]),
        )

    def __len__(self):
        return int(self.sample_offsets[-1])

    def scenario_index_range(self, scenario_id):
        return int(self.sample_offsets[scenario_id]), int(self.sample_offsets[scenario_id + 1])

    def _read_count_series(self, csv_path):
        df = pd.read_csv(csv_path)
        counts = np.zeros((len(df), self.num_disc), dtype=np.float32)
        for i, name in enumerate(self.disc_names):
            if name in df.columns:
                counts[:, i] = df[name].to_numpy(dtype=np.float32)
        return counts

    def _get_counts_norm(self, scenario_id):
        if self._cache_scenario_id == scenario_id:
            return self._cache_counts_norm

        scenario = self.scenarios[scenario_id]
        logger.debug("Loading scenario %d from %s", scenario_id, scenario["path"])
        counts = self._read_count_series(scenario["path"])
        counts_norm = counts / float(scenario["N"])

        self._cache_scenario_id = scenario_id
        self._cache_counts_norm = counts_norm
        return counts_norm

    def build_graph_data(self, reaction_probs, N, current_counts_norm, delta):
        x = torch.zeros((self.num_nodes, self.feature_dim), dtype=torch.float32)

        current_counts_norm = torch.as_tensor(current_counts_norm, dtype=torch.float32)
        reaction_probs = torch.as_tensor(reaction_probs, dtype=torch.float32)
        N_norm = float(N) / self.max_N

        x[: self.num_disc, 0] = current_counts_norm
        x[: self.num_disc, 1] = self.mass_tensor
        x[: self.num_disc, 2] = self.radius_tensor
        x[: self.num_disc, 4] = 1.0
        x[: self.num_disc, 6] = N_norm

        x[self.num_disc :, 3] = reaction_probs
        x[self.num_disc :, 5] = 1.0
        x[self.num_disc :, 6] = N_norm

        y = torch.zeros(self.num_nodes, dtype=torch.float32)
        y[: self.num_disc] = torch.as_tensor(delta, dtype=torch.float32)

        return Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=y,
            loss_mask=self.loss_mask,
        )

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        scenario_id = int(np.searchsorted(self.sample_offsets, idx, side="right") - 1)
        local_t = int(idx - self.sample_offsets[scenario_id])

        scenario = self.scenarios[scenario_id]
        counts_norm = self._get_counts_norm(scenario_id)

        curr = counts_norm[local_t]
        nxt = counts_norm[local_t + 1]
        delta = nxt - curr

        return self.build_graph_data(
            reaction_probs=scenario["reaction_probs"],
            N=scenario["N"],
            current_counts_norm=curr,
            delta=delta,
        )


class ScenarioBatchSampler(Sampler):
    """
    Yields batches from one scenario at a time.
    This keeps file loads/cache reuse efficient.
    """

    def __init__(self, dataset, scenario_ids, batch_size, shuffle_scenarios=False, shuffle_within=False, seed=42):
        self.dataset = dataset
        self.scenario_ids = list(scenario_ids)
        self.batch_size = batch_size
        self.shuffle_scenarios = shuffle_scenarios
        self.shuffle_within = shuffle_within
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        scenario_ids = self.scenario_ids.copy()

        if self.shuffle_scenarios:
            rng.shuffle(scenario_ids)

        for scenario_id in scenario_ids:
            start, end = self.dataset.scenario_index_range(scenario_id)
            if end <= start:
                continue

            idxs = np.arange(start, end, dtype=np.int64)
            if self.shuffle_within:
                rng.shuffle(idxs)

            for i in range(0, len(idxs), self.batch_size):
                yield idxs[i : i + self.batch_size].tolist()

    def __len__(self):
        total = 0
        for scenario_id in self.scenario_ids:
            start, end = self.dataset.scenario_index_range(scenario_id)
            n = max(end - start, 0)
            total += (n + self.batch_size - 1) // self.batch_size
        return total


def split_scenarios(dataset, train_ratio=0.8, seed=42):
    scenario_ids = list(range(len(dataset.scenarios)))
    rng = np.random.default_rng(seed)
    rng.shuffle(scenario_ids)

    n_train = max(1, int(len(scenario_ids) * train_ratio))
    train_ids = scenario_ids[:n_train]
    val_ids = scenario_ids[n_train:]

    if len(val_ids) == 0:
        val_ids = train_ids[:1]

    logger.info(
        "Scenario split: %d train | %d val | train_ratio=%.3f",
        len(train_ids),
        len(val_ids),
        train_ratio,
    )

    return train_ids, val_ids


class MLP(nn.Module):
    def __init__(self, dims, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CountGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=4, edge_dim=2, dropout=0.0):
        super().__init__()
        self.node_encoder = MLP([in_dim, hidden_dim, hidden_dim], dropout=dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            nn_edge = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn_edge, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.decoder = MLP([hidden_dim, hidden_dim, 1], dropout=dropout)

    def forward(self, data):
        h = self.node_encoder(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + conv(h, data.edge_index, data.edge_attr)
            h = norm(h)
            h = F.relu(h)
        delta = self.decoder(h).squeeze(-1)
        return delta


def train_one_epoch(model, loader, optimizer, device, epoch=None, log_interval_batches=0):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        loss = F.smooth_l1_loss(pred[batch.loss_mask], batch.y[batch.loss_mask])

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

        if log_interval_batches > 0 and (batch_idx % log_interval_batches == 0 or batch_idx == num_batches):
            logger.info(
                "Epoch %03d | train batch %d/%d | running_loss=%.6f",
                epoch if epoch is not None else -1,
                batch_idx,
                num_batches,
                total_loss / max(total_graphs, 1),
            )

    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device, epoch=None, log_interval_batches=0):
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        pred = model(batch)
        loss = F.smooth_l1_loss(pred[batch.loss_mask], batch.y[batch.loss_mask])

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

        if log_interval_batches > 0 and (batch_idx % log_interval_batches == 0 or batch_idx == num_batches):
            logger.info(
                "Epoch %03d | val batch %d/%d | running_loss=%.6f",
                epoch if epoch is not None else -1,
                batch_idx,
                num_batches,
                total_loss / max(total_graphs, 1),
            )

    return total_loss / max(total_graphs, 1)


def save_checkpoint(path, model, meta, model_kwargs):
    ckpt = {
        "state_dict": model.state_dict(),
        "meta": meta,
        "model_kwargs": model_kwargs,
    }
    torch.save(ckpt, path)
    logger.info("Checkpoint saved to %s", path)


def load_trained_model(checkpoint_path, device="cpu"):
    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = CountGNN(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info("Checkpoint loaded successfully")
    return model, ckpt["meta"]


def build_graph_from_meta(meta, reaction_probs, N, current_counts_norm):
    disc_static = meta["disc_static"]
    disc_names = [d["name"] for d in disc_static]
    disc_index = {name: i for i, name in enumerate(disc_names)}
    reaction_cols = meta["reaction_cols"]
    max_N = float(meta["max_N"])
    mass_scale = float(meta["mass_scale"])
    radius_scale = float(meta["radius_scale"])
    feature_dim = int(meta["feature_dim"])

    num_disc = len(disc_names)
    num_rxn = len(reaction_cols)
    num_nodes = num_disc + num_rxn

    x = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)
    N_norm = float(N) / max_N

    for i, d in enumerate(disc_static):
        x[i, 0] = float(current_counts_norm[i])
        x[i, 1] = float(d["mass"]) / mass_scale
        x[i, 2] = float(d["radius"]) / radius_scale
        x[i, 4] = 1.0
        x[i, 6] = N_norm

    edge_src, edge_dst, edge_attr = [], [], []

    for j, rstr in enumerate(reaction_cols):
        node_idx = num_disc + j
        x[node_idx, 3] = float(reaction_probs.get(rstr, 0.0))
        x[node_idx, 5] = 1.0
        x[node_idx, 6] = N_norm

        reactants, products = parse_reaction(rstr)

        for name in reactants:
            d_idx = disc_index[name]
            edge_src += [d_idx, node_idx]
            edge_dst += [node_idx, d_idx]
            edge_attr += [[1.0, 0.0], [1.0, 0.0]]

        for name in products:
            d_idx = disc_index[name]
            edge_src += [node_idx, d_idx]
            edge_dst += [d_idx, node_idx]
            edge_attr += [[0.0, 1.0], [0.0, 1.0]]

    if len(edge_src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        eattr = torch.empty((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        eattr = torch.tensor(edge_attr, dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=eattr,
        loss_mask=torch.tensor([True] * num_disc + [False] * num_rxn, dtype=torch.bool),
    )


@torch.no_grad()
def rollout_counts(
    model,
    meta,
    reaction_probs,
    N,
    steps,
    initial_counts=None,
    initial_fractions=None,
    device="cpu",
    round_to_int=False,
):
    disc_names = meta["disc_names"]
    num_disc = len(disc_names)
    dt = float(meta["dt"])

    if initial_counts is None and initial_fractions is None:
        raise ValueError("Provide either `initial_counts` or `initial_fractions`.")

    if initial_counts is None:
        initial_counts = np.array([float(initial_fractions.get(name, 0.0)) * N for name in disc_names], dtype=float)
    else:
        initial_counts = np.array([float(initial_counts.get(name, 0.0)) for name in disc_names], dtype=float)

    current_counts_norm = initial_counts / max(float(N), 1.0)

    logger.info("Starting rollout for %d steps", steps)

    rows = []
    rows.append([0.0] + initial_counts.tolist())

    for step in range(steps):
        data = build_graph_from_meta(
            meta=meta,
            reaction_probs=reaction_probs,
            N=N,
            current_counts_norm=current_counts_norm,
        ).to(device)

        pred_delta = model(data).detach().cpu().numpy()
        current_counts_norm = current_counts_norm + pred_delta[:num_disc]
        current_counts_norm = np.clip(current_counts_norm, 0.0, None)

        counts = current_counts_norm * float(N)
        if round_to_int:
            counts = np.rint(counts)

        rows.append([(step + 1) * dt] + counts.tolist())

    columns = ["ElapsedTime[s]"] + disc_names
    logger.info("Rollout complete")
    return pd.DataFrame(rows, columns=columns)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_csv", type=str, default="DATASETS_DIR/WDN/simple_transformation_set_2.csv")
    parser.add_argument("--series_dir", type=str, default="DATASETS_DIR/WDN/simple_transformation_set_2")
    parser.add_argument("--config_json", type=str, default="CONFIG_DIR/WDN/transformation_simple.json")
    parser.add_argument("--out", type=str, default="count_gnn.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--log_interval_batches", type=int, default=50)
    parser.add_argument("--log_interval_scenarios", type=int, default=50)
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Arguments: %s", vars(args))

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    logger.info("Creating dataset...")
    dataset = ReactionCountDataset(
        mapping_csv=args.mapping_csv,
        series_dir=args.series_dir,
        config_json=args.config_json,
        log_interval_scenarios=args.log_interval_scenarios,
    )
    logger.info("Dataset ready: %d total samples", len(dataset))

    train_scenarios, val_scenarios = split_scenarios(
        dataset,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    train_sampler = ScenarioBatchSampler(
        dataset,
        train_scenarios,
        batch_size=args.batch_size,
        shuffle_scenarios=True,
        shuffle_within=True,
        seed=args.seed,
    )
    val_sampler = ScenarioBatchSampler(
        dataset,
        val_scenarios,
        batch_size=args.batch_size,
        shuffle_scenarios=False,
        shuffle_within=False,
        seed=args.seed,
    )

    logger.info(
        "Sampler batches: %d train | %d val",
        len(train_sampler),
        len(val_sampler),
    )

    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model_kwargs = {
        "in_dim": dataset.feature_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "edge_dim": 2,
        "dropout": args.dropout,
    }
    model = CountGNN(**model_kwargs).to(device)

    logger.info(
        "Model initialized: hidden_dim=%d | num_layers=%d | trainable_params=%d",
        args.hidden_dim,
        args.num_layers,
        count_parameters(model),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger.info("Optimizer: AdamW | lr=%g | weight_decay=%g", args.lr, args.weight_decay)

    best_val = float("inf")
    best_state = None

    logger.info("Starting training for %d epochs...", args.epochs)

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        logger.info("Epoch %03d started", epoch)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            log_interval_batches=args.log_interval_batches,
        )
        val_loss = eval_one_epoch(
            model,
            val_loader,
            device,
            epoch=epoch,
            log_interval_batches=0,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("Epoch %03d produced new best val loss: %.6f", epoch, val_loss)

        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

    if best_state is not None:
        logger.info("Restoring best model state with val loss %.6f", best_val)
        model.load_state_dict(best_state)

    save_checkpoint(args.out, model, dataset.meta, model_kwargs)
    print(f"Saved model to: {args.out}")


if __name__ == "__main__":
    main()
