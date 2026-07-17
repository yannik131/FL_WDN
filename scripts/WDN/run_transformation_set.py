import json
from util.paths import CONFIG_DIR, DATASETS_DIR
from util.task import Task, execute_tasks, create_mapfile
from itertools import product

# Reduced to ~288 scenarios (~5.7M samples total) for fast training on RTX 3070
p_vals = [0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.9]
A0_vals = [0, 10, 80, 320, 1000]
B0_vals = A0_vals.copy()

mapping = {
    "p": ["config", "reactions", 0, "probability"],
    "N": ["config", "cellMembraneType", "discCount"],
    "f_A": ["config", "cellMembraneType", "discTypeDistribution", "A"],
    "f_B": ["config", "cellMembraneType", "discTypeDistribution", "B"]
}

tasks = []
i = 0
for p, A0, B0 in product(p_vals, A0_vals, B0_vals):
    for r in range(3):  # Reduced from 5 to 3 repetitions
        N = A0 + B0
        params = dict(
            filename=f"simple_transformation_set_3_{i:04d}.csv",
            r=r,
            p=p,
            N=N,
            f_A=A0/N if N > 0 else 1,
            f_B=B0/N if N > 0 else 0
        )
        tasks.append(Task(params, mapping))
        i += 1

mapfile_path = DATASETS_DIR / "WDN/simple_transformation_set_3.csv"
create_mapfile(tasks, mapfile_path)

with open(CONFIG_DIR / "WDN/transformation_simple.json") as f:
    cfg = json.load(f)

output_dir = DATASETS_DIR / "WDN/simple_transformation_set_3"

if __name__ == "__main__":
    execute_tasks(tasks, cfg, output_dir)
