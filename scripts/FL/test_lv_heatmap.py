import pandas as pd 
from pathlib import Path 
from lv_classifier import has_lv_dynamics
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = Path(__file__).resolve().parent
mapping = pd.read_csv(script_dir / "../datasets/FL/simple_lv_set.txt")

rows = []
for p1 in np.linspace(0, 1, 101):
    for p2 in np.linspace(0, 1, 101):
        for _ in range(4):
            is_lv = p1 * p2 > p2 / p1  * np.random.random()
            rows.append((p1, p2, int(is_lv)))

df_all = pd.DataFrame(rows, columns=["p1", "p2", "is_lv"])
heat = df_all.groupby(["p1", "p2"])["is_lv"].mean().reset_index()
pivot = heat.pivot(index="p1", columns="p2", values="is_lv") * 100

sns.heatmap(pivot, cmap="viridis", vmin=0, vmax=100)
plt.show()
