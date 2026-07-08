import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# Generate Latin hypercube samples
sampler = qmc.LatinHypercube(d=6, seed=42)
samples = sampler.random(n=20000)

lower = np.array([0, 0.0, 0.0, 0.8, 0, 0])
upper = np.array([1.0, 0.2, 0.15, 1.0, 0.11, 0.11])

samples = qmc.scale(samples, lower, upper)

print("Total 6D combinations:", len(samples))

names = ["p1", "p2", "p3", "p4", "p5", "p6"]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.ravel()

for i, ax in enumerate(axes):
    values = samples[:, i]

    counts, bins, patches = ax.hist(
        values,
        bins=50,
        edgecolor="black"
    )

    unique_count = len(np.unique(values))

    ax.set_title(
        f"{names[i]}: {unique_count} unique values"
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

    ax.axvline(lower[i], linestyle="--")
    ax.axvline(upper[i], linestyle="--")

plt.tight_layout()
plt.show()