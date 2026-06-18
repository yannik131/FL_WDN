import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

HERE = Path(__file__).resolve().parent

df = pd.read_csv(HERE / "../datasets/example/good.csv")
t = df["ElapsedTime[s]"].to_numpy(dtype=float)
x = df["Prey"].to_numpy(dtype=float)
y = df["Predator"].to_numpy(dtype=float)

eps = 1e-12

# -----------------------
# TUNABLE THRESHOLDS
# -----------------------
SMOOTH_FRAC = 0.05
MIN_WIN = 7

PREY_PROM_FRAC = 0.35      # prey peak prominence >= PREY_PROM_FRAC * std(prey)
PRED_PROM_FRAC = 0.35      # predator peak prominence >= PRED_PROM_FRAC * std(predator)
MIN_PEAK_DISTANCE_FRAC = 0.08   # minimum distance between peaks, as fraction of trace length

MIN_LAG_FRAC = 0.05        # predator peak must occur at least 5% into prey->prey cycle
MAX_LAG_FRAC = 0.80        # predator peak must occur before 80% of that cycle
MIN_GOOD_PAIR_FRACTION = 0.80
REQUIRE_EXACTLY_ONE_PREDATOR_PEAK_PER_CYCLE = True

# -----------------------
# HELPERS
# -----------------------
def odd_window(n, frac=0.05, min_win=7):
    w = max(min_win, int(frac * n))
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - 1 if n % 2 == 0 else n
    return max(5, w)

# -----------------------
# SMOOTHING
# -----------------------
win = odd_window(len(t), frac=SMOOTH_FRAC, min_win=MIN_WIN)
poly = min(3, win - 1)

x_s = savgol_filter(x, win, poly)
y_s = savgol_filter(y, win, poly)

# -----------------------
# PEAK FINDING
# -----------------------
dist = max(1, int(MIN_PEAK_DISTANCE_FRAC * len(t)))

prey_peaks, prey_props = find_peaks(
    x_s,
    prominence=PREY_PROM_FRAC * np.std(x_s),
    distance=dist
)

pred_peaks, pred_props = find_peaks(
    y_s,
    prominence=PRED_PROM_FRAC * np.std(y_s),
    distance=dist
)

if len(prey_peaks) < 2:
    print("NO (not enough prey peaks to define cycles)")
    raise SystemExit

prey_prom = dict(zip(prey_peaks, prey_props["prominences"]))
pred_prom = dict(zip(pred_peaks, pred_props["prominences"]))

# -----------------------
# MATCH PEAKS BY CYCLE
# One prey cycle = between consecutive prey peaks
# -----------------------
pairs = []

for i in range(len(prey_peaks) - 1):
    p0 = prey_peaks[i]
    p1 = prey_peaks[i + 1]

    cycle_start = t[p0]
    cycle_end = t[p1]
    cycle_len = cycle_end - cycle_start

    candidates = pred_peaks[(pred_peaks > p0) & (pred_peaks < p1)]

    row = {
        "prey_idx": int(p0),
        "prey_time": float(t[p0]),
        "prey_prom": float(prey_prom[p0]),
        "cycle_len": float(cycle_len),
        "n_pred_in_cycle": int(len(candidates)),
        "good": False,
        "reason": ""
    }

    if len(candidates) == 0:
        row["reason"] = "no predator peak after prey peak in this cycle"
        pairs.append(row)
        continue

    if REQUIRE_EXACTLY_ONE_PREDATOR_PEAK_PER_CYCLE and len(candidates) != 1:
        row["reason"] = f"{len(candidates)} predator peaks in cycle"
        pairs.append(row)
        continue

    q = candidates[0]
    lag = t[q] - t[p0]
    lag_frac = lag / (cycle_len + eps)

    row.update({
        "pred_idx": int(q),
        "pred_time": float(t[q]),
        "pred_prom": float(pred_prom[q]),
        "lag": float(lag),
        "lag_frac": float(lag_frac),
    })

    if lag <= 0:
        row["reason"] = "predator peak does not lag prey peak"
    elif lag_frac < MIN_LAG_FRAC:
        row["reason"] = "lag too small"
    elif lag_frac > MAX_LAG_FRAC:
        row["reason"] = "lag too large"
    else:
        row["good"] = True
        row["reason"] = "ok"

    pairs.append(row)

good_pairs = [r for r in pairs if r["good"]]
good_fraction = len(good_pairs) / max(len(pairs), 1)

is_predator_lagged_oscillatory = (
    len(good_pairs) >= 1 and
    good_fraction >= MIN_GOOD_PAIR_FRACTION
)

# -----------------------
# REPORT
# -----------------------
print(f"prey peaks found: {len(prey_peaks)}")
print(f"predator peaks found: {len(pred_peaks)}")
print(f"good peak pairs: {len(good_pairs)} / {len(pairs)}")
print(f"good pair fraction: {good_fraction:.3f}")

if good_pairs:
    lags = np.array([r["lag"] for r in good_pairs], dtype=float)
    lag_fracs = np.array([r["lag_frac"] for r in good_pairs], dtype=float)
    print(f"mean lag: {np.mean(lags):.3f} s")
    print(f"mean lag fraction of cycle: {np.mean(lag_fracs):.3f}")

for i, r in enumerate(pairs, start=1):
    msg = f"cycle {i}: prey@{r['prey_time']:.3f}s"
    if "pred_time" in r:
        msg += (
            f", pred@{r['pred_time']:.3f}s"
            f", lag={r['lag']:.3f}s"
            f", lag_frac={r['lag_frac']:.3f}"
            f", result={r['reason']}"
        )
    else:
        msg += f", result={r['reason']}"
    print(msg)

print(
    "YES (clear predator-lagged peak pairs)"
    if is_predator_lagged_oscillatory
    else "NO (peak pattern not clean enough)"
)

# -----------------------
# PLOT
# -----------------------
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(t, x, alpha=0.25, color="green", label="prey raw")
ax.plot(t, y, alpha=0.25, color="red", label="predator raw")
ax.plot(t, x_s, color="green", label="prey smooth")
ax.plot(t, y_s, color="red", label="predator smooth")

ax.plot(t[prey_peaks], x_s[prey_peaks], "o", color="darkgreen", label="prey peaks")
ax.plot(t[pred_peaks], y_s[pred_peaks], "o", color="darkred", label="predator peaks")

for r in good_pairs:
    ax.plot(
        [r["prey_time"], r["pred_time"]],
        [x_s[r["prey_idx"]], y_s[r["pred_idx"]]],
        "--",
        color="gray",
        alpha=0.8
    )

ax.set_title("Peak-pair check")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Population")
ax.legend()
plt.tight_layout()
plt.show()
