import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
import numpy as np
from lv_classifier import has_lv_dynamics

fig, axes = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(10, 4))

HERE = Path(__file__).parent

show_legend = True
for ax, filename in zip(axes, ["medium.csv", "good.csv"]):
    print(filename)
    df = pd.read_csv(HERE / "../../datasets/FL/example" / filename)
    df = df.drop(columns=['Resource'])
    W = 500
    t, prey, predator = df['ElapsedTime[s]'], savgol_filter(df['Prey'], W, 3), savgol_filter(df['Predator'], W, 3)
    OK = has_lv_dynamics(df)
    ax.plot(df['ElapsedTime[s]'], prey, label='Prey', color='tab:green')
    ax.plot(df['ElapsedTime[s]'], predator, label='Predator', color='tab:red')
    if show_legend:
        ax.legend()
        show_legend = False

    prey_peaks, _ = find_peaks(prey, prominence=150)
    predator_peaks, _ = find_peaks(predator, prominence=120)
    df.plot(x="ElapsedTime[s]", ax=ax, color={'Prey': 'tab:green', 'Predator': 'tab:red'}, legend=False, alpha=0.5)
    ax.set_ylabel("N")

    ax.scatter(
        t.iloc[prey_peaks],
        prey[prey_peaks],
        color="green",
        s=40,
        alpha=0.5,
        zorder=5,
        label="Prey peaks"
    )

    ax.scatter(
        t.iloc[predator_peaks],
        predator[predator_peaks],
        color="red",
        s=40,
        alpha=0.5,
        zorder=5,
        label="Predator peaks"
    )

    ax.set_title(f"Has LV dynamics: {'Yes' if OK else 'No'}")

from matplotlib.ticker import MultipleLocator

axes[0].yaxis.set_major_locator(MultipleLocator(300))

plt.tight_layout()
plt.savefig(HERE / "../../reports/FL/figures/lv_good_bad.jpg", dpi=300)