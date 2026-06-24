import json
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from bin_selector import get_binary_path
import os

script_dir = Path(__file__).resolve().parent
exe = get_binary_path()

with open(script_dir / "../config/WDN/transformation_simple.json") as f:
    base_cfg = json.load(f)

p_vals = np.linspace(0, 1, 101)
tasks = [float(p) for p in p_vals]
output_dir = script_dir / "../datasets/WDN/simple_transformation_set/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

def run_sim(p):
    cfg = json.loads(json.dumps(base_cfg))  # cheap deep copy
    cfg["config"]["reactions"][0]["probability"] = p
    out = f"{output_dir}/{p:.2f}.csv"

    if Path(out).exists():
        print(f"Skipping {out} since it exists")
        return out

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name

    subprocess.run(
        [
            str(exe),
            f"--config={tmp_path}",
            f"--out={out}",
            "--duration=60",
            "--storage-interval=0.003",
        ],
        cwd=script_dir,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return out


if __name__ == "__main__":
    workers = os.cpu_count()
    print(f"Number of workers: {workers}")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_sim, t) for t in tasks]

        with tqdm(total=len(tasks)) as pbar:
            for _ in as_completed(futures):
                pbar.update(1)
