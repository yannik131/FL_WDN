import matplotlib.pyplot as plt
import pandas as pd
from util.paths import DATASETS_DIR, RESULTS_DIR

plt.figure(figsize=(10, 4))
df = pd.read_csv(DATASETS_DIR / "FL/example/good.csv")
x = df["ElapsedTime[s]"]
colors = {
    "Resource": "tab:blue",
    "Prey": "tab:green",
    "Predator": "tab:red"
}

for col in df.columns:
    if col not in colors:
        continue
    plt.plot(x, df[col], label=col, color=colors[col])

plt.xlabel("t [s]")
plt.ylabel("N")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "FL/example_good.png", dpi=300)