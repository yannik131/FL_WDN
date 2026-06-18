import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent

df = pd.read_csv(HERE / "../datasets/example/good.csv")
t = df["ElapsedTime[s]"].values
x = df["Prey"].values
y = df["Predator"].values

# --- smoothing ---
dt = np.mean(np.diff(t))
win = int(0.02 * len(t)) | 1
x_s = savgol_filter(x, win, 3)
y_s = savgol_filter(y, win, 3)

# --- derivatives ---
dx = np.gradient(x_s, dt)
dy = np.gradient(y_s, dt)

eps = 1e-8
x_safe = np.maximum(x_s, eps)
y_safe = np.maximum(y_s, eps)

# --- fit LV parameters (linear regression form) ---
A = np.column_stack([x_safe, -x_safe * y_safe])
α, β = np.linalg.lstsq(A, dx, rcond=None)[0]

B = np.column_stack([x_safe * y_safe, -y_safe])
δ, γ = np.linalg.lstsq(B, dy, rcond=None)[0]

# --- stability fix: clip parameters ---
α, β, γ, δ = np.clip([α, β, γ, δ], -5.0, 5.0)

# --- LV model ---
def lv(t, z):
    x, y = z
    return [
        α * x - β * x * y,
        δ * x * y - γ * y
    ]

z0 = [x_s[0], y_s[0]]

# --- solve with stiff-capable solver + safety controls ---
sol = solve_ivp(
    lv,
    (t[0], t[-1]),
    z0,
    t_eval=t,
    method="LSODA",
    rtol=1e-5,
    atol=1e-7,
    max_step=dt * 5
)

if (not sol.success) or np.any(~np.isfinite(sol.y)):
    print("NO (integration failed)")
    exit()

x_hat, y_hat = sol.y

# --- error metric ---
err = np.mean((x_s - x_hat)**2 + (y_s - y_hat)**2)
scale = np.mean(x_s**2 + y_s**2)
rel_err = err / (scale + eps)

THRESHOLD = 0.05
is_lv = rel_err < THRESHOLD

print("alpha, beta, gamma, delta =", (α, β, γ, δ))
print("relative error =", rel_err)
print("YES (LV dynamics)" if is_lv else "NO (not LV dynamics)")

# =======================
# PLOTS
# =======================

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# --- original data ---
axes[0].plot(t, x, alpha=0.3, color="green", label="prey raw")
axes[0].plot(t, y, alpha=0.3, color="red", label="predator raw")
axes[0].plot(t, x_s, color="green", label="prey smooth")
axes[0].plot(t, y_s, color="red", label="predator smooth")
axes[0].set_title("Original data")
axes[0].legend()

# --- fit vs data ---
axes[1].plot(t, x_s, color="green", label="prey data")
axes[1].plot(t, y_s, color="red", label="predator data")
axes[1].plot(t, x_hat, "--", color="green", label="prey LV fit")
axes[1].plot(t, y_hat, "--", color="red", label="predator LV fit")
axes[1].set_title("Lotka–Volterra fit")
axes[1].legend()

# --- phase space ---
axes[2].plot(x_s, y_s, color="black", label="data")
axes[2].plot(x_hat, y_hat, "--", color="blue", label="LV model")
axes[2].set_title("Phase space")
axes[2].set_xlabel("Prey")
axes[2].set_ylabel("Predator")
axes[2].legend()

plt.tight_layout()
plt.show()