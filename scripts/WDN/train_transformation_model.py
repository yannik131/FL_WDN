from pathlib import Path
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import joblib
import matplotlib.pyplot as plt
from util.paths import DATASETS_DIR, MODELS_DIR, RESULTS_DIR

DATA_DIR = DATASETS_DIR / "WDN/simple_transformation_set/"
MAP_FILE = DATASETS_DIR / "WDN/simple_transformation_set.txt"
MODEL_OUT = MODELS_DIR / "WDN/simple_transformation_mlp.joblib"
RESULT_OUT = RESULTS_DIR / "WDN/simple_transformation/ex2/"
T_MAX = 60
T_GRID = np.linspace(0, T_MAX, 301)
RANDOM_STATE = 0

def load_mean_trajectories():
    trajectories_file_pickled = DATASETS_DIR / "WDN/simple_transformation_trajectories.pkl"
    if trajectories_file_pickled.exists():
        with open(trajectories_file_pickled, "rb") as f:
            return pickle.load(f)

    mapping = pd.read_csv(MAP_FILE)
    grouped = defaultdict(list)

    for row in tqdm(mapping.itertuples(), total=len(mapping)):
        csv_path = DATA_DIR / row.Filename
        df = pd.read_csv(csv_path)

        t = df["ElapsedTime[s]"].to_numpy(dtype=float)
        A = df["A"].to_numpy(dtype=float)

        A0 = int(df.iloc[0]["A"])
        B0 = int(df.iloc[0]["B"])
        p = float(row.p)

        A_interp = np.interp(T_GRID, t, A)
        grouped[(A0, B0, p)].append(A_interp)

    trajectories = []
    for (A0, B0, p), runs in grouped.items():
        A_mean = np.mean(np.stack(runs, axis=0), axis=0)
        trajectories.append(
            {
                "A0": A0,
                "B0": B0,
                "p": p,
                "A_mean": A_mean
            }
        )

    with open(trajectories_file_pickled, "wb") as f:
        pickle.dump(trajectories, f)

    return trajectories

def build_dataset(trajectories):
    X = []
    y = []
    group_ids = []

    for gid, tr in enumerate(trajectories):
        A0 = tr["A0"]
        B0 = tr["B0"]
        p = tr["p"]
        N = A0 + B0

        if N == 0:
            continue

        A0_frac = A0 / N
        B0_frac = B0 / N
        A_frac_series = tr["A_mean"] / N

        for t, a_frac in zip(T_GRID, A_frac_series):
            X.append([A0_frac, B0_frac, np.log1p(N), p, t / T_MAX])
            y.append(a_frac)
            group_ids.append(gid)

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(group_ids, dtype=int)
    )

def train_model(X, y, group_ids):
    unique_groups = np.unique(group_ids)
    train_groups, test_groups = train_test_split(
        unique_groups, test_size=0.2, random_state=RANDOM_STATE
    )

    train_mask = np.isin(group_ids, train_groups)
    test_mask = np.isin(group_ids, test_groups)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s = x_scaler.transform(X_test)

    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE,
        verbose=True
    )
    model.fit(X_train_s, y_train_s)

    y_pred_s = model.predict(X_test_s)
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE on A/n: {mse:.6e}")

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }

def predict_series(bundle, A0, B0, p, t_grid=None, round_counts=False):
    if t_grid is None:
        t_grid = T_GRID

    N = A0 + B0
    if N == 0:
        return pd.DataFrame(
            {"ElapsedTime[s]": t_grid, "A": np.zeros_like(t_grid), "B": np.zeros_like(t_grid)}
        )

    X = np.column_stack(
        [
            np.full_like(t_grid, A0 / N, dtype=float),
            np.full_like(t_grid, B0 / N, dtype=float),
            np.full_like(t_grid, np.log1p(N), dtype=float),
            np.full_like(t_grid, p, dtype=float),
            t_grid / T_MAX
        ]
    )

    Xs = bundle["x_scaler"].transform(X)
    A_frac_s = bundle["model"].predict(Xs)
    A_frac = bundle["y_scaler"].inverse_transform(A_frac_s.reshape(-1, 1)).ravel()

    A = np.clip(A_frac * N, 0.0, float(N))
    B = N - A

    if round_counts:
        A = np.rint(A).astype(int)
        B = N - A

    return pd.DataFrame(
        {
            "ElapsedTime[s]": t_grid,
            "A": A,
            "B": B
        }
    )

if __name__ == "__main__":
    if MODEL_OUT.exists():
        bundle = joblib.load(MODEL_OUT)
    else:
        trajectories = load_mean_trajectories()
        X, y, group_ids = build_dataset(trajectories)
        bundle = train_model(X, y, group_ids)
        joblib.dump(bundle, MODEL_OUT)

    A0 = 10
    B0 = 0
    p = 0.01
    pred = predict_series(bundle, A0=A0, B0=B0, p=p, round_counts=True)
    pred.to_csv(RESULT_OUT / f"pred_{A0}_{B0}_{p}.csv", index=False)
    pred.plot(x="ElapsedTime[s]")
    plt.title(f"Predicted, A0={A0}, B0={B0}, p={p}")
    plt.show()
