import pandas as pd
from .lv_classifier import has_lv_dynamics
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from util.paths import DATASETS_DIR, RESULTS_DIR
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import numpy as np

heatmap_df_path = DATASETS_DIR / "FL/lv_heat_map_simple_df.csv"

def process_row(row):
    df = pd.read_csv(DATASETS_DIR / f"FL/simple_lv_set/{row.Filename}")
    is_lv = has_lv_dynamics(df)
    return (row.p1, row.p2, int(is_lv))

if not heatmap_df_path.exists():
    mapping = pd.read_csv(DATASETS_DIR / "FL/simple_lv_set.txt")
    rows_iter = list(mapping.itertuples(index=True))
    rows = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(process_row, r) for r in rows_iter]

        for f in tqdm(as_completed(futures), total=len(futures)):
            rows.append(f.result())

    df_all = pd.DataFrame(rows, columns=["p1", "p2", "is_lv"])
    df_all.to_csv(DATASETS_DIR / "FL/lv_heat_map_simple_df.csv", index=False)
else:
    df_all = pd.read_csv(heatmap_df_path)

heat = df_all.groupby(["p1", "p2"])["is_lv"].mean().reset_index()
pivot = heat.pivot(index="p1", columns="p2", values="is_lv") * 100

heat.to_csv(DATASETS_DIR / "FL/lv_heat_map_pivot.csv", index=False)

model = joblib.load(RESULTS_DIR / "FL/simple_lv_model.joblib")
grid = heat[["p1", "p2"]].copy()
grid["lv_prob_model"] = model.predict_proba(grid[["p1", "p2"]])[:, 1]
pivot_model = grid.pivot(index="p1", columns="p2", values="lv_prob_model") * 100

fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 8))
sns.heatmap(pivot, cmap="viridis", vmin=0, vmax=100, square=True, ax=axes[0], cbar=False)
axes[0].set_title("Simulated")
sns.heatmap(pivot_model, cmap="viridis", vmin=0, vmax=100, square=True, ax=axes[1])
axes[1].set_title("Predicted")

ticks = np.round(np.arange(0, 1.01, 0.05), 2)
tick_labels = [f"{t:.2f}" for t in ticks]

pos = np.linspace(0, len(pivot.columns) - 1, len(ticks))

for ax in axes:
    ax.set_xticks(pos)
    ax.set_yticks(pos)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

plt.savefig(RESULTS_DIR / "FL/simple_comp.png", dpi=300)

unique_probs = np.sort(grid["lv_prob_model"].round(2).unique())
print(unique_probs)