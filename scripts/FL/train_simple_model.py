import pandas as pd
import numpy as np
from util.paths import DATASETS_DIR, RESULTS_DIR
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import joblib

df = pd.read_csv(DATASETS_DIR / "FL/lv_heat_map_simple_df.csv")
groups = pd.factorize(list(zip(df["p1"], df["p2"])))[0]

X = df[["p1", "p2"]].to_numpy()
y = df["is_lv"].astype(int).to_numpy()

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

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

print("ROC AUC: ", roc_auc_score(y_test, p_test))

joblib.dump(model, RESULTS_DIR / "FL/simple_lv_model.joblib")