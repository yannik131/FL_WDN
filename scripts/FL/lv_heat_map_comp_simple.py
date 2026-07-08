import pandas as pd
from .lv_classifier import has_lv_dynamics
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from util.paths import DATASETS_DIR, RESULTS_DIR
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import numpy as np
import matplotlib.ticker as mtick

heatmap_df_path = DATASETS_DIR / "FL/lv_heat_map_simple_prom_150_120.csv"

def process_row(row):
    df = pd.read_csv(DATASETS_DIR / f"FL/simple_lv_set/{row.Filename}")
    is_lv = has_lv_dynamics(df)
    return (row.p1, row.p2, int(is_lv))


def set_ticks(ax, n):
    ticks = np.arange(0, 1.01, 0.05)
    tick_labels = [f"{t:.2f}" for t in ticks]
    pos = ticks * (n - 1)
    ax.set_xticks(pos)
    ax.set_yticks(pos)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)


if not heatmap_df_path.exists():
    mapping = pd.read_csv(DATASETS_DIR / "FL/simple_lv_set.txt")
    rows_iter = list(mapping.itertuples(index=True))
    rows = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(process_row, r) for r in rows_iter]

        for f in tqdm(as_completed(futures), total=len(futures)):
            rows.append(f.result())

    df_all = pd.DataFrame(rows, columns=["p1", "p2", "is_lv"])
    df_all.to_csv(heatmap_df_path, index=False)
else:
    df_all = pd.read_csv(heatmap_df_path)

heat = df_all.groupby(["p1", "p2"])["is_lv"].mean().reset_index()
pivot = heat.pivot(index="p1", columns="p2", values="is_lv") * 100

model = joblib.load(RESULTS_DIR / "FL/simple_lv_model.joblib")

# 1000 x 1000 prediction grid: 0.000, 0.001, ..., 0.999
p = np.arange(0, 1, 0.001)
p1_grid, p2_grid = np.meshgrid(p, p, indexing="ij")

grid_model = pd.DataFrame({
    "p1": p1_grid.ravel(),
    "p2": p2_grid.ravel(),
})
grid_model["lv_prob_model"] = model.predict_proba(grid_model[["p1", "p2"]])[:, 1]

pivot_model = grid_model.pivot(index="p1", columns="p2", values="lv_prob_model") * 100

fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 6))

sns.heatmap(
    pivot,
    cmap="viridis",
    vmin=0,
    vmax=100,
    square=True,
    ax=axes[0],
    cbar=False,
)
axes[0].set_title("Simulated")

sns.heatmap(
    pivot_model,
    cmap="viridis",
    vmin=0,
    vmax=100,
    square=True,
    ax=axes[1],
    cbar_kws={'format': mtick.PercentFormatter(100)}
)
axes[1].set_title("Predicted")

set_ticks(axes[0], len(pivot.columns))
set_ticks(axes[1], len(pivot_model.columns))

plt.savefig(RESULTS_DIR / "FL/full_comp.jpg", dpi=300)

unique_probs = np.sort(grid_model["lv_prob_model"].round(2).unique())
print(unique_probs)
