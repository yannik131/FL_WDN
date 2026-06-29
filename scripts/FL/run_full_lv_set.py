from scipy.stats import qmc 
from util.task import Task, execute_tasks, create_mapfile
from util.paths import DATASETS_DIR, CONFIG_DIR
import json

sampler = qmc.LatinHypercube(d=6, seed=42)
samples = sampler.random(n=20000)

mapping = {
    f"p{j}": ["config", "reactions", j - 1, "probability"]
    for j in range(1, 7)
}

tasks = []
i = 0
for sample in samples:
    for r in range(5):
        params = dict(filename=f"full_lv_set_{i:04d}.csv", r=r)
        for j in range(1, 7):
            params[f"p{j}"] = sample[j - 1]
        tasks.append(Task(params, mapping))
        i += 1

mapfile_path = DATASETS_DIR / "FL/full_lv_set.csv"
create_mapfile(tasks, mapfile_path)

output_dir = DATASETS_DIR / "FL/full_lv_set/"
output_dir.mkdir(parents=True, exist_ok=True)
with open(CONFIG_DIR / "FL/preyPredator.json") as f:
    cfg = json.load(f)

if __name__ == "__main__":
    execute_tasks(tasks, cfg, output_dir)