import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
import numpy as np

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)

HERE = Path(__file__).parent

for ax, filename in zip(axes, ["bad.csv", "medium.csv", "good.csv"]):
    print(filename)
    df = pd.read_csv(HERE / "../datasets/FL/example" / filename)
    df = df.drop(columns=['Resource'])
    W = 500
    t, prey, predator = df['ElapsedTime[s]'], savgol_filter(df['Prey'], W, 3), savgol_filter(df['Predator'], W, 3)
    ax.plot(df['ElapsedTime[s]'], prey, label='Prey', color='tab:green')
    ax.plot(df['ElapsedTime[s]'], predator, label='Predator', color='tab:red')
    prey_peaks, _ = find_peaks(prey, prominence=0.05*np.median(prey), height=1.1*np.median(prey))
    predator_peaks, _ = find_peaks(predator, prominence=0.05*np.median(predator), height=1.1*np.median(predator))
    df.plot(x="ElapsedTime[s]", ax=ax, color={'Prey': 'tab:green', 'Predator': 'tab:red'}, legend=True, alpha=0.3)
    ax.axhline(np.median(prey), color='green', linestyle='--', alpha=0.5)
    ax.axhline(np.median(predator), color='red', linestyle='--', alpha=0.5)

    for prey_x in t.iloc[prey_peaks]:
        ax.axvline(prey_x, color="green", linestyle="--", alpha=0.5)
    for predator_x in t.iloc[predator_peaks]:
        ax.axvline(predator_x, color="red", linestyle="--", alpha=0.5)

    prey_t = t.iloc[prey_peaks].to_numpy()
    predator_t = t.iloc[predator_peaks].to_numpy()
    nearest_peaks = []
    peak_pair_t = set()
    for pt in prey_t:
        idx = np.argmin(np.abs(predator_t - pt))
        nearest_peaks.append(predator_t[idx])
        peak_pair_t.add(pt)
        peak_pair_t.add(predator_t[idx])

    cond_pairs = len(peak_pair_t) >= 6
    print(f"{len(peak_pair_t)} unique prey/predator peak x values: {'OK' if cond_pairs else 'NOT OK'}")

    nearest_peaks = np.array(nearest_peaks)

    delays = nearest_peaks - prey_t
    cond_lag = np.median(delays) > 0
    print(f"Median lag = {np.median(delays)}: {'OK' if cond_lag else 'NOT OK'}")

    prey_periods = np.diff(prey_t)
    predator_periods = np.diff(predator_t)

    prey_T = np.mean(prey_periods) if len(prey_periods) else np.nan
    predator_T = np.mean(predator_periods) if len(predator_periods) else np.nan

    def check_periods(peaks, T):
        if len(peaks) < 2:
            return False
        print(f"{np.abs(np.diff(peaks))} -> {T}")
        return np.all(np.abs(np.diff(peaks) - T) <= 0.5 * T)

    cond_preys = check_periods(prey_t, prey_T)
    cond_preds = check_periods(predator_t, predator_T)

    ok = cond_pairs and cond_lag and cond_preds

    ax.set_title(f"{filename}: {'YES' if ok else 'NO'}")

plt.tight_layout()
plt.show()