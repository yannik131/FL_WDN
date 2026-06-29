from util.paths import DATASETS_DIR
import pandas as pd 
import matplotlib.pyplot as plt

mapping = pd.read_csv(DATASETS_DIR / "FL/simple_lv_set.txt")
p1 = 0.14
p2 = 0.14
files = mapping[(mapping["p1"] == p1) & (mapping["p2"] == p2)]
fig, axes = plt.subplots(4, 1)

for i, file in enumerate(files["Filename"]):
    ax = axes[i]
    df = pd.read_csv(DATASETS_DIR / "FL/simple_lv_set" / file).drop(columns=["Resource"])
    df.plot(x="ElapsedTime[s]", ax = ax)
    ax.set_title(file)

plt.show()
