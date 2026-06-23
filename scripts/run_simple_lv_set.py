import json
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

script_dir = Path(__file__).resolve().parent
exe = script_dir.parent / "bin/cell-cli"

with open(script_dir / "../config/Fl/preyPredator.json") as f:
    base_cfg = json.load(f)

p1_vals = np.linspace(0, 1, 101)
p2_vals = np.linspace(0, 1, 101)

tasks = [(float(p1), float(p2)) for p1 in p1_vals for p2 in p2_vals]


def run_sim(p1_p2):
    p1, p2 = p1_p2

    cfg = json.loads(json.dumps(base_cfg))  # cheap deep copy

    cfg["config"]["reactions"][0]["probability"] = p1
    cfg["config"]["reactions"][1]["probability"] = p2

    out = f"../datasets/FL/simple_lv_set/{p1:.2f}_{p2:.2f}.csv"

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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    return out


if __name__ == "__main__":
    workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_sim, t) for t in tasks]

        with tqdm(total=len(tasks)) as pbar:
            for _ in as_completed(futures):
                pbar.update(1)