import pandas as pd
import matplotlib.pyplot as plt
import glob
import re

files = sorted(glob.glob("[0-9]*.csv"), key=lambda x: int(re.findall(r"\d+", x)[0]))

fig, axes = plt.subplots(len(files), 1, sharex=True, figsize=(8, 2 * len(files)))

if len(files) == 1:
    axes = [axes]

for ax, f in zip(axes, files):
    df = pd.read_csv(f)
    df.plot(x="ElapsedTime[s]", ax=ax, legend=False)
    ax.set_title(f)

plt.tight_layout()
plt.show()