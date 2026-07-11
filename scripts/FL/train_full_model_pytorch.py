import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from util.paths import DATASETS_DIR

# Binary classifier with 6 input nodes, 2 hidden layers with 64 nodes each, and 1 output node
# layer 1: 6*64 + 64
# layer 2: 64*64 + 64
# layer 3: 64*1 + 1
# = 4673 parameters
# Reasonable default for arbitrary, non-linear 6D shapes
# Output: The value z of the output node is a logit
# we could put nn.Sigmoid() after it to obtain a probability via p = 1/(1 + exp(-z)) and use BCELoss() directly later
# but BCEWithLogitsLoss() is numerically more stable and thus preferred
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

df = pd.read_csv(DATASETS_DIR / "FL/lv_heat_map_full_3.csv")
probs = ["p1","p2","p3","p4","p5","p6"]

# [[p1, ..., p6], [p1, ..., p6], ...]
X = df[probs].to_numpy()

# [0, 1, ...]
y = df["is_lv"].astype(np.float32).to_numpy()

# assigns ascending numbers to each unique X vector
# [1, 2, 3, 2, 3, 2, ...]
groups = pd.factorize(
    df[probs].apply(tuple, axis=1)
)[0]

# k: We want to split all probability vectors into 80% training and 20% testing -> n_splits = 5 so each group contains 20% of the data
# stratified: each split contains about the same number of positive and negative results (1/0 in y)
# group: since each X vector occurs 5 times (simulation was run 5 times), we want every run to be in the same dataset to avoid leakage: the test dataset shouldn't see things from the training dataset

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# store folds 2-5 in train and fold 1 in test
train_idx, test_idx = next(sgkf.split(X, y, groups))
np.save(DATASETS_DIR / "FL/train_idx_3.npy", train_idx)
np.save(DATASETS_DIR / "FL/test_idx_3.npy", test_idx)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# scale each p_i column to have zero mean and unit variance
# this way most values are between -2 and +2
# this avoids problems with too small/large weights since the probabilities are unequally distributed:
# p1 is in [0, 1] whereas p5 for example is in [0, 0.1]
# train the scaler on the training data (80%)
# use the same scaler for the test data to again avoid leaking information from train to test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert to pytorch tensor format
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test, dtype=torch.float32)

# each gradient update uses 256 samples for updating
# they are all shuffled to avoid batches with only a single class
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=256, shuffle=True
)

model = MLP()

# binary cross entropy, standard for binary classification to calculate the loss (how far the models guess is from the actual training data)
criterion = nn.BCEWithLogitsLoss()

# use a smart gradient descent algorithm for updating the parameters
# learning rate of 0.001 is a common default to avoid large jumps
# weight_decay penalizes large weights to avoid overfitting
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

for epoch in range(40):
    print(epoch)
    # xb, yb contains the batch of 256 samples from the training data
    for xb, yb in train_loader:
        # clear accumulated gradients
        optimizer.zero_grad()
        # calculate the bce loss with logits from the model
        loss = criterion(model(xb), yb)
        # calculate the new gradient and store it in the model parameters
        loss.backward()
        # update the parameters with the gradient
        optimizer.step()

    with torch.no_grad():
        # calculate the logits with the model for the test dataset
        logits = model(X_test)
        # convert logits to probabilities
        p_test = torch.sigmoid(logits).numpy()

    # brier measures spread of predictions around the true outcomes
    # brier = 1/N * sum (p_i - y_i)^2
    # perfect model has 0, worst has 1
    print("Brier score:", brier_score_loss(y_test, p_test))

    # log loss measures the same thing as brier but heavily punishes false negatives and false positives
    # log_loss = -1/N sum (y_i log(p_i) + (1-y_i)*log(1-p_i))
    # if true label is 1, the loss is -log(p) -> p small (confident false prediction) -> high loss
    # if true label is 0, the loss is -log(1-p) -> p close to one (confident true prediction) -> high loss
    # perfect model has 0, worst has very large loss value
    print("Log loss:", log_loss(y_test, p_test))

    # auc measures how well the decision boundary was found
    # whereas log loss/brier measure how accurate the model is within and outside that boundary
    # perfect is 1, worst is 0.5
    print("ROC AUC:", roc_auc_score(y_test, p_test))

    # example output for dataset 3 after 40 epochs:
    # Brier score: 0.028854428011338036
    # Log loss: 0.09529701088378399
    # ROC AUC: 0.9673378460055462
    # -> very good!