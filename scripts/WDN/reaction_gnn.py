print("Starting imports")
import logging
from WDN.reaction_dataset import load_dataset, create_graph
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from util.paths import RESULTS_DIR
from time import perf_counter_ns
print("Done importing")

logger = logging.getLogger('mylogger')

class ReactionGNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Message from a reactant species to its reaction node.
        # Input: [reactant_fraction, mass, reaction_probability]
        self.reactant_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Message from reaction node to product species.
        # Input: [reaction_hidden_state, product1_fraction, product2_fraction, reaction_probability]
        self.product_mlp = nn.Sequential(
            nn.Linear(35, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        x = data.x
        educt1 = x[data.educt1_indices]
        educt2 = x[data.educt2_indices]
        educt1_mass = data.masses[data.educt1_indices]
        educt2_mass = data.masses[data.educt2_indices]
        product1 = x[data.product1_indices]
        product2 = x[data.product2_indices]
        educt_input1 = torch.cat([educt1, educt1_mass, data.reaction_probs], dim=-1)
        educt_input2 = torch.cat([educt2, educt2_mass, data.reaction_probs], dim=-1)
        msg1 = self.reactant_mlp(educt_input1)
        msg2 = self.reactant_mlp(educt_input2)
        reaction_messages = msg1 + data.has_2_educts * msg2
        product_input = torch.cat([
            reaction_messages,
            product1,
            product2 * data.has_2_products,
            data.reaction_probs
        ], dim=-1)
        predicted_fraction = torch.sigmoid(self.product_mlp(product_input))
        educt_amount = educt1 * (
            (1.0 - data.has_2_educts)
            + data.has_2_educts * educt2
        )
        requested_flux = predicted_fraction * data.reaction_probs * educt_amount
        requested_outgoing = torch.zeros_like(x)
        requested_outgoing.index_add_(0, data.educt1_indices, requested_flux)
        requested_outgoing.index_add_(0, data.educt2_indices, requested_flux * data.has_2_educts)
        species_scale = torch.clamp(x / requested_outgoing.clamp_min(1e-12), max=1.0)
        scale1 = species_scale[data.educt1_indices]
        scale2 = torch.where(
            data.has_2_educts.bool(),
            species_scale[data.educt2_indices],
            torch.ones_like(scale1)
        )
        flux = requested_flux * torch.minimum(scale1, scale2)
        outgoing = torch.zeros_like(x)
        outgoing.index_add_(0, data.educt1_indices, flux)
        outgoing.index_add_(0, data.educt2_indices, flux * data.has_2_educts)
        incoming = torch.zeros_like(x)
        incoming.index_add_(0, data.product1_indices, flux)
        incoming.index_add_(0, data.product2_indices, flux * data.has_2_products)
        return x - outgoing + incoming

def rollout_loss(model, batch, loss_fn, rollout_steps=20):
    state = batch.x
    loss = 0.0
    for step in range(rollout_steps):
        batch.x = state
        state = model(batch)
        if not torch.isfinite(state).all():
            raise RuntimeError(f'Non-finite prediction at rollout step {step}')

        target = batch.y[:, step:step + 1]

        loss = loss + loss_fn(state, target)
        if not torch.isfinite(loss):
            raise RuntimeError('Loss is non-finite')

    return loss / rollout_steps

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            loss = rollout_loss(model, batch, loss_fn)
            total_loss += loss.item()

    return total_loss / len(loader)

def train(set_name, device="cpu", epochs=None):
    train_dataset, test_dataset = load_dataset()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = ReactionGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"Number of batches: {len(train_loader)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()

            loss = rollout_loss(model, batch, loss_fn)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.6g}")

    model_path = RESULTS_DIR / f"WDN/{set_name}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss:.6g}")

def load_model(set_name, device="cpu"):
    model_path = RESULTS_DIR / f"WDN/{set_name}.pt"
    model = ReactionGNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_trajectory(
    model,
    initial_species_values,
    masses,
    reactions,
    steps,
    dt=0.05,
    device="cpu",
):
    values = list(map(float, initial_species_values))
    trajectory = [(0.0, *values)]

    print("Starting prediction")
    start = perf_counter_ns()
    with torch.no_grad():
        for step in range(1, steps + 1):
            graph = create_graph(values, masses, reactions).to(device)
            pred = model(graph)

            values = pred.squeeze(-1).cpu().tolist()
            trajectory.append((step * dt, *values))
    dt = (perf_counter_ns() - start) / 1e9
    print(f"Prediction took {dt:.3g}s")

    return trajectory


