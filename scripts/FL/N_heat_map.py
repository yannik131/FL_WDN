import numpy as np
import matplotlib.pyplot as plt

eps = 1e-9
d = 0.15
N = np.arange(0, 21)
prey, pred = np.meshgrid(N, N)

Z = np.abs(prey - pred) / (prey + pred + eps)

plt.figure(figsize=(6, 5))

im = plt.imshow(Z, origin='lower', extent=[0, 20, 0, 20], cmap='viridis')

plt.contour(pred, prey, Z, levels=[d], colors='white', linewidths=2)

plt.colorbar(im, label='|N_prey - N_pred| / (N_prey + N_pred)')

plt.xlabel('N_pred')
plt.ylabel('N_prey')
plt.title(f'Peak imbalance with d ≤ {d} contour')

plt.tight_layout()
plt.show()