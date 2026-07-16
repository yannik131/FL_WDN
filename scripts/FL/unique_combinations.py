import pandas as pd
from util.paths import DATASETS_DIR

df = pd.read_csv(DATASETS_DIR / "FL/lv_heat_map_full_3.csv")

count = (
    df[df["is_lv"] == 1]
    .drop_duplicates(subset=["p1", "p2", "p3", "p4", "p5", "p6"])
    .shape[0]
)

print(count)