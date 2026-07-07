import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from util.paths import DATASETS_DIR, RESULTS_DIR
import matplotlib.ticker as mtick
import numpy as np

df1_path = DATASETS_DIR / "FL/lv_heat_map_simple_prom_50.csv"
df2_path = DATASETS_DIR / "FL/lv_heat_map_simple_prom_150_120.csv"

def set_ticks(ax, n):
    ticks = np.arange(0, 1.01, 0.05)
    tick_labels = [f"{t:.2f}" for t in ticks]
    pos = ticks * (n - 1)
    ax.set_xticks(pos)
    ax.set_yticks(pos)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

heat1 = pd.read_csv(df1_path).groupby(["p1", "p2"])["is_lv"].mean().reset_index()
heat2 = pd.read_csv(df2_path).groupby(["p1", "p2"])["is_lv"].mean().reset_index()

pivot1 = heat1.pivot(index="p1", columns="p2", values="is_lv") * 100
pivot2 = heat2.pivot(index="p1", columns="p2", values="is_lv") * 100

fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 6))

sns.heatmap(
    pivot1,
    cmap="viridis",
    vmin=0,
    vmax=100,
    square=True,
    ax=axes[0],
    cbar=False,
    cbar_kws={'format': mtick.PercentFormatter(100)}
)
axes[0].set_title("Prominence prey/predator: 50/50")

sns.heatmap(
    pivot2,
    cmap="viridis",
    vmin=0,
    vmax=100,
    square=True,
    ax=axes[1],
    cbar_kws={'format': mtick.PercentFormatter(100)}
)
axes[1].set_title("Prominence prey/predator: 150/120")

set_ticks(axes[0], len(pivot1.columns))
set_ticks(axes[1], len(pivot2.columns))

plt.savefig(RESULTS_DIR / "FL/simple_prom_comp.jpg", dpi=300)
