import pandas as pd 
from pathlib import Path 
from lv_classifier import has_lv_dynamics
from tqdm import tqdm 
import seaborn as sns
import matplotlib.pyplot as plt


script_dir = Path(__file__).resolve().parent
heatmap_df_path = script_dir / "../datasets/FL/lv_heat_map_simple_df.csv"

if not heatmap_df_path.exists():
    mapping = pd.read_csv(script_dir / "../datasets/FL/simple_lv_set.txt")

    rows = []
    for row in tqdm(mapping.itertuples(index=True), total=len(mapping)):
        df = pd.read_csv(script_dir / f"../datasets/FL/simple_lv_set/{row.Filename}")
        is_lv = has_lv_dynamics(df)
        rows.append((row.p1, row.p2, int(is_lv)))

    df_all = pd.DataFrame(rows, columns=["p1", "p2", "is_lv"])
    df_all.to_csv(script_dir / "../datasets/FL/lv_heat_map_simple_df.csv", index=False)
else:
    df_all = pd.read_csv(heatmap_df_path)

heat = df_all.groupby(["p1", "p2"])["is_lv"].mean().reset_index()
pivot = heat.pivot(index="p1", columns="p2", values="is_lv") * 100

heat.to_csv(script_dir / "../datasets/FL/lv_heat_map_pivot.csv", index=False)

sns.heatmap(pivot, cmap="viridis", vmin=0, vmax=100, square=True)
plt.show()