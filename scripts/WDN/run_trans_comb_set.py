import json
from util.paths import CONFIG_DIR, DATASETS_DIR
from util.task import Task, execute_tasks, create_mapfile
from itertools import product

set_name = "trans_comp_set_1"
trans_p_vals = [0.0, 0.01, 0.05, 0.1, 0.5, 0.9]
comb_p_vals = trans_p_vals.copy()
A0_vals = [0, 10, 100, 500]
B0_vals = A0_vals.copy()
C0_vals = A0_vals.copy()

mapping = {
    "p1": ["config", "reactions", 0, "probability"],
    "p2": ["config", "reactions", 1, "probability"],
    "N": ["config", "cellMembraneType", "discCount"],
    "f_A": ["config", "cellMembraneType", "discTypeDistribution", "A"],
    "f_B": ["config", "cellMembraneType", "discTypeDistribution", "B"],
    "f_C": ["config", "cellMembraneType", "discTypeDistribution", "C"]
}

tasks = []
i = 0
for p1, p2, A0, B0, C0 in product(trans_p_vals, comb_p_vals, A0_vals, B0_vals, C0_vals):
    N = A0 + B0 + C0
    params = dict(
        filename=f"{set_name}_{i:04d}.csv",
        r=1,
        p1=p1,
        p2=p2,
        N=N,
        f_A=A0/N if N > 0 else 1,
        f_B=B0/N if N > 0 else 0,
        f_C=C0/N if N > 0 else 0
    )
    tasks.append(Task(params, mapping))
    i += 1

mapfile_path = DATASETS_DIR / f"WDN/{set_name}.csv"
create_mapfile(tasks, mapfile_path, header_mapping={'p1': 'A->B', 'p2': 'A+B->C'})

with open(CONFIG_DIR / f"WDN/trans_comp.json") as f:
    cfg = json.load(f)

output_dir = DATASETS_DIR / f"WDN/{set_name}"

if __name__ == "__main__":
    execute_tasks(tasks, cfg, output_dir)
