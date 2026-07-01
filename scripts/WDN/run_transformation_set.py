import json
from util.paths import CONFIG_DIR, DATASETS_DIR
from util.task import Task, execute_tasks, create_mapfile
from itertools import product

p_vals = [0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.3, 0.5, 0.8, 0.9, 0.999]
A0_vals = [0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1000]
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
    for r in range(5):
        N = A0 + B0
        params = dict(
            filename=f"simple_transformation_set_2_{i:04d}.csv",
            r=r,
            p=p,
            N=N,
            f_A=A0/N if N > 0 else 1,
            f_B=B0/N if N > 0 else 0
        )
        tasks.append(Task(params, mapping))
        i += 1

mapfile_path = DATASETS_DIR / "WDN/simple_transformation_set_2.csv"
create_mapfile(tasks, mapfile_path)

with open(CONFIG_DIR / "WDN/transformation_simple.json") as f:
    cfg = json.load(f)

output_dir = DATASETS_DIR / "WDN/simple_transformation_set_2"

if __name__ == "__main__":
    execute_tasks(tasks, cfg, output_dir)