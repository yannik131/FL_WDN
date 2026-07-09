import pandas as pd
import numpy as np
from util.paths import DATASETS_DIR, RESULTS_DIR
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import joblib

df = pd.read_csv(DATASETS_DIR / "FL/lv_heat_map_full_3.csv")
groups = pd.factorize(list(zip(df["p1"], df["p2"], df["p3"], df["p4"], df["p5"], df["p6"])))[0]

X = df[["p1", "p2", "p3", "p4", "p5", "p6"]].to_numpy()
y = df["is_lv"].astype(int).to_numpy()

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)
train_idx, test_idx = next(gss.split(X, y, groups=groups))


X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("True cases in test set: ", np.unique(y_test, return_counts=True))

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        alpha=1e-3,
        max_iter=500,
        early_stopping=True,
        random_state=42
    ))
])

model.fit(X_train, y_train)

p_test = model.predict_proba(X_test)[:, 1]

print("Brier score: ", brier_score_loss(y_test, p_test))
print("Log loss: ", log_loss(y_test, p_test))
print("ROC AUC: ", roc_auc_score(y_test, p_test))

joblib.dump(model, RESULTS_DIR / "FL/full_lv_model.joblib")