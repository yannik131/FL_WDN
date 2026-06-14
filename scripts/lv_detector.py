import pandas as pd 
import numpy as np 
from pathlib import Path 
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter, find_peaks, correlate, correlation_lags

HERE = Path(__file__).resolve().parent 

df = pd.read_csv(HERE / "../datasets/example/bad.csv")
t, prey, predator = df["ElapsedTime[s]"], df['Prey'], df['Predator']

fig, axes = plt.subplots(2, 1)
df.drop(columns=["Resource"]).plot(x="ElapsedTime[s]", ax=axes[0])

dt = 0.003
T = 60.0
total = 1000.0

W = int(T / dt * 0.02) | 1
prom = 0.05 * total

prey = savgol_filter(prey, W, 3)
predator = savgol_filter(predator, W, 3)

px, _ = find_peaks(prey, prominence=0.01*prey.max())
py, _ = find_peaks(predator, prominence=0.01*predator.max())

distances = []
unique_predator_peaks = set()

for p in px:
    nearest_predator_peak = py[np.argmin(np.abs(p - py))]
    unique_predator_peaks.add(nearest_predator_peak)
    distances.append(t.iloc[nearest_predator_peak] - t.iloc[p])
    print(f"Prey peak at {t.iloc[p]}, {prey[p]}: Nearest predator peak at {t.iloc[nearest_predator_peak]}, {predator[nearest_predator_peak]}")

median = np.median(distances)
print(median)
print(f"{int(len(unique_predator_peaks) / len(px) * 100.0)}% match")

for x in t.iloc[px]:
    axes[1].axvline(x, color="green", linestyle="--", alpha=0.5)

for x in t.iloc[py]:
    axes[1].axvline(x, color="red", linestyle="--", alpha=0.5)

axes[1].plot(t, prey, label="prey", color="green")
axes[1].plot(t, predator, label="predator", color="red")

plt.legend()
plt.show()