import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="+", help="One or more CSV files")
args = parser.parse_args()

files = args.csv

if len(files) == 1:
    df = pd.read_csv(files[0])
    df.plot(x="ElapsedTime[s]", legend=True)
    plt.title(Path(files[0]).name)
    plt.show()

else:
    n = len(files)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True, constrained_layout=True)

    if n == 1:
        axes = [axes]

    for ax, f in zip(axes, files):
        df = pd.read_csv(f)
        df.plot(x="ElapsedTime[s]", legend=True, ax=ax)
        ax.set_title(Path(f).name)

    plt.show()