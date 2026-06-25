import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

ROOT_DIR = Path(__file__).parent.parent.parent
RESULT_DIR = ROOT_DIR / "results/WDN/simple_transformation/ex2"

files = list(RESULT_DIR.glob("*.csv"))

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

ax_dict = {10: axes[0], 100: axes[1], 1000: axes[2]}

# fixed colors per signal
color_map = {
    "A": "tab:blue",
    "B": "tab:orange",
}

for file in files:
    df = pd.read_csv(file)
    filename = file.stem

    label = "Predicted"
    linestyle = "--"  # default
    if filename.startswith("sim"):
        label = "Simulated"
        linestyle = "-"

    m = re.search(r"\D+_(\d+)_(\d+)_(\d+\.\d+)", filename)
    if not m:
        raise RuntimeError("Invalid filename: " + filename)

    A0 = int(m.group(1))
    B0 = int(m.group(2))
    p = float(m.group(3))

    ax = ax_dict[A0]
    ax.set_title(f"A0={A0}, B0={B0}, p={p}")

    for col in df.columns:
        if col == "ElapsedTime[s]":
            continue
        if col not in color_map:
            continue
        ax.plot(
            df["ElapsedTime[s]"],
            df[col],
            label=f"{col} ({label})",
            color=color_map[col],
            linestyle=linestyle,
        )

for ax in axes:
    ax.legend()

plt.tight_layout()
plt.show()