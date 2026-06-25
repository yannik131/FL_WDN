import tkinter as tk
from tkinter import ttk
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import savgol_filter, find_peaks


HERE = Path(__file__).parent
FILES = ["bad.csv", "medium.csv", "good.csv"]


def compute_and_plot(ax, filename, prominence):
    df = pd.read_csv(HERE / "../datasets/FL/example" / filename)
    df = df.drop(columns=["Resource"])

    W = 500
    t = df["ElapsedTime[s]"]
    prey = savgol_filter(df["Prey"], W, 3)
    predator = savgol_filter(df["Predator"], W, 3)

    ax.clear()

    ax.plot(t, prey, label="Prey", color="tab:green")
    ax.plot(t, predator, label="Predator", color="tab:red")

    prey_peaks, _ = find_peaks(prey, prominence=prominence)
    predator_peaks, _ = find_peaks(predator, prominence=prominence)

    ax.axhline(np.median(prey), color="green", linestyle="--", alpha=0.5)
    ax.axhline(np.median(predator), color="red", linestyle="--", alpha=0.5)

    for x in t.iloc[prey_peaks]:
        ax.axvline(x, color="green", linestyle="--", alpha=0.4)
    for x in t.iloc[predator_peaks]:
        ax.axvline(x, color="red", linestyle="--", alpha=0.4)

    prey_t = t.iloc[prey_peaks].to_numpy()
    predator_t = t.iloc[predator_peaks].to_numpy()

    nearest = []
    peak_pair_t = set()

    for pt in prey_t:
        if len(predator_t) == 0:
            continue
        idx = np.argmin(np.abs(predator_t - pt))
        nearest.append(predator_t[idx])
        peak_pair_t.add(pt)
        peak_pair_t.add(predator_t[idx])

    cond_pairs = len(peak_pair_t) >= 6

    if len(prey_t) and len(predator_t):
        delays = np.array(nearest) - prey_t[:len(nearest)]
        cond_lag = np.median(delays) > 0
    else:
        cond_lag = False

    prey_periods = np.diff(prey_t)
    predator_periods = np.diff(predator_t)

    prey_T = np.mean(prey_periods) if len(prey_periods) else np.nan
    predator_T = np.mean(predator_periods) if len(predator_periods) else np.nan

    def check_periods(peaks, T):
        if len(peaks) < 2:
            return False
        return np.all(np.abs(np.diff(peaks) - T) <= 0.5 * T)

    cond_preys = check_periods(prey_t, prey_T)
    cond_preds = check_periods(predator_t, predator_T)

    ok = cond_pairs and cond_lag and cond_preds

    ax.set_title(f"{filename}: {'YES' if ok else 'NO'}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Peak Prominence Explorer")

        self.prominence = tk.DoubleVar(value=40)

        control = ttk.Frame(self)
        control.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control, text="Prominence").pack(side=tk.LEFT)
        self.spin = ttk.Spinbox(
            control,
            from_=0,
            to=500,
            increment=1,
            textvariable=self.prominence,
            command=self.redraw,
            width=10,
        )
        self.spin.pack(side=tk.LEFT)

        self.fig, self.axes = plt.subplots(3, 1, sharex=True, sharey=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.redraw()

    def redraw(self):
        p = float(self.prominence.get())
        for ax, f in zip(self.axes, FILES):
            compute_and_plot(ax, f, p)
        self.fig.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    App().mainloop()
