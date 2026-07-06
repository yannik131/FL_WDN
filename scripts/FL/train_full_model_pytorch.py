import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from util.paths import DATASETS_DIR
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, in_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

df = pd.read_csv(DATASETS_DIR / "FL/lv_heat_map_full_2_df.csv")
X = df[["p1","p2","p3","p4","p5","p6"]].to_numpy()
y = df["is_lv"].astype(np.float32).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=256, shuffle=True
)

model = MLP()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

for epoch in range(5):
    print(epoch)
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X_test)
        p_test = torch.sigmoid(logits).numpy()

    print("Brier score:", brier_score_loss(y_test, p_test))
    print("Log loss:", log_loss(y_test, p_test))
    print("ROC AUC:", roc_auc_score(y_test, p_test))