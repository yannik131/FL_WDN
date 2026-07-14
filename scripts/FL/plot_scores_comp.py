import pandas as pd
import matplotlib.pyplot as plt
from util.paths import RESULTS_DIR

# Load data
df = pd.read_csv(RESULTS_DIR / "FL/scores.csv")

baseline = {
    "brier": 0.029208642136701157,
    "log": 0.09578933691123727,
    "auc": 0.9669857416782169,
}

scores = ["brier", "log", "auc"]
colors = {
    "brier": "tab:blue",
    "log": "tab:orange",
    "auc": "tab:green",
}

plt.figure(figsize=(8, 5))

for score in scores:
    color = colors[score]

    # Baseline
    plt.axhline(
        baseline[score],
        color=color,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )

    # split_random = 0
    d0 = df[df["split_random"] == 0].sort_values("n_clients")
    plt.plot(
        d0["n_clients"],
        d0[score],
        color=color,
        linestyle="-",
        marker="o",
        label=f"{score} (cohesive split)",
    )

    # split_random = 1
    d1 = df[df["split_random"] == 1].sort_values("n_clients")
    plt.plot(
        d1["n_clients"],
        d1[score],
        color=color,
        linestyle=":",
        marker="s",
        label=f"{score} (random split)",
    )

plt.xscale("log", base=2)
plt.xticks(sorted(df["n_clients"].unique()), sorted(df["n_clients"].unique()))
plt.xlabel("Number of clients")
plt.ylabel("Score")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "FL/scores_comp.jpg", dpi=300)