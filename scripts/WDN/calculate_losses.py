import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from util.paths import RESULTS_DIR, DATASETS_DIR
from WDN.reaction_gnn import ReactionGNN, evaluate

print("Loading model")
test_dataset = torch.load(DATASETS_DIR / f"WDN/full_set_4_test.pt", weights_only=False)
print("Done loading")
for model_name in ["full_set_3", "full_set_4_5_epochs", "full_set_4_10_epochs"]:
    print(f"Calculating loss for model: {model_name}")
    model = ReactionGNN().to("cpu")
    loss_fn = nn.MSELoss()
    model_path = RESULTS_DIR / f"WDN/{model_name}.pt"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_loss = evaluate(model, test_loader, loss_fn, "cpu")
    print(f"Test loss: {test_loss:.6g}")