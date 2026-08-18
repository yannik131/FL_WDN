from WDN.reaction_gnn import predict_trajectory, train, load_model
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from WDN.resample_counts import resample_counts
from util.paths import RESULTS_DIR
from matplotlib.lines import Line2D

def water_example(model, device="cpu"):
    species = ["H+", "H20", "Ca²⁺", "CO2g", "CO2aq", "CO₃²⁻", "HCO3-", "H2CO3", "CaCO₃ (s)"]
    initial_fractions = [0, 0.8, 0.1, 0.1, 0, 0, 0, 0, 0]

    masses = [1, 18, 40,   44,  44,   60,   61,   62,   100]
    #         H+ H20 Ca2+  CO2g CO2aq CO32- HCO3- H2CO3 CaCO3
    #         0  1   2     3    4     5     6     7     8
    reactions = [
        [[[3], [4]], 0.05],
        [[[4], [3]], 0.01],
        [[[4, 1], [7]], 0.02],
        [[[7], [4, 1]], 0.2],
        [[[7], [0, 6]], 0.865],
        [[[0, 6], [7]], 0.12],
        [[[6], [0, 5]], 0.03],
        [[[0, 5], [6]], 0.3],
        [[[8], [2, 5]], 0.002],
        [[[2, 5], [8]], 0.015],
    ]

    trajectory = predict_trajectory(
        model=model,
        initial_species_values=initial_fractions,
        masses=masses,
        reactions=reactions,
        steps=int(1000 / 0.05),
        dt=0.05,
        device=device,
    )
    trajectory = np.array(trajectory)
    fig, ax = plt.subplots()

    colors = ['blue', 'red', 'green']
    df = pd.read_csv(RESULTS_DIR / "WDN/water_averaged_df.csv")
    df = resample_counts(df)

    for color, name in zip(colors, ['Ca²⁺', 'CO₃²⁻', 'CaCO₃ (s)']):
        i = species.index(name)
        ax.plot(trajectory[:, 0], trajectory[:, i+1], linestyle="--", color=color)
        ax.plot(trajectory[:, 0], df[name].to_numpy(), color=color, label=name)

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color='black', linestyle='-', label='averaged'),
        Line2D([0], [0], color='black', linestyle='--', label='predicted')
    ]

    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend(handles=handles)
    plt.tight_layout()
    plt.show()

def lv_example(model, devie):
    species = ["Prey", "Predator", "Resource"]
    initial_fractions = [0, 0, 1]

    masses = [30,  30,      30]
    #         Prey Predator Resource
    #         0    1        2
    reactions = [
        [[[0, 1], [1, 1]], 0.5],
        [[[0, 2], [0, 0]], 0.02],
        [[[0], [2]], 0.05],
        [[[1], [2]], 0.9],
        [[[2], [0]], 0.01],
        [[[2], [1]], 0.01]
    ]

    trajectory = predict_trajectory(
        model=model,
        initial_species_values=initial_fractions,
        masses=masses,
        reactions=reactions,
        steps=int(120 / 0.05),
        dt=0.05,
        device=device,
    )
    trajectory = np.array(trajectory)

    for name in species:
        i = species.index(name)
        plt.plot(trajectory[:, 0], trajectory[:, i+1], label=name)

    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = "cpu" # used to prefer cuda, but turned out to be slower
    set_name = "full_set_3"
    print(f"Using device: {device}")
    model_path = RESULTS_DIR / f"WDN/{set_name}.pt"

    if not model_path.exists():
        train(set_name, device=device, epochs=5)
        exit(0)

    model = load_model(set_name)
    water_example(model)
    